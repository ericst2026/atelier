"""Step 3 — the training loop. Relaunches under torchrun when more than one GPU is allocated."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atelier_sdk import Result, inputs, params, parse_args, progress  # noqa: E402

parse_args()
P = params({"max_iters": 2000, "batch_size": 32, "grad_accum": 1, "lr": 6e-4, "warmup_iters": 100, "weight_decay": 0.1, "eval_interval": 100, "eval_iters": 20, "dtype": "bf16", "compile": False, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
n_gpus = int(os.environ.get("ATELIER_GPUS", "1") or 1)

if n_gpus > 1 and "RANK" not in os.environ:
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={n_gpus}", __file__, "--run-dir", str(run_dir)]
    print("[train] relaunching:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lib.model import GPT, GPTConfig  # noqa: E402

ddp = "RANK" in os.environ
if ddp:
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
else:
    rank, world, device = 0, 1, ("cuda" if torch.cuda.is_available() else "cpu")
master = rank == 0
torch.manual_seed(int(P["seed"]) + rank)

meta = json.loads(Path(I["meta"]).read_text())
np_dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
train_data = np.memmap(I["train_bin"], dtype=np_dtype, mode="r")
val_data = np.memmap(I["val_bin"], dtype=np_dtype, mode="r")
cfg = GPTConfig(**json.loads(Path(I["model_config"]).read_text()))
B, T = int(P["batch_size"]), cfg.block_size


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - T - 1, (B,))
    x = torch.stack([torch.from_numpy(d[i : i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1 : i + 1 + T].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


model = GPT(cfg).to(device)
raw_model = model
if bool(P["compile"]):
    model = torch.compile(model)
if ddp:
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(model, device_ids=[local_rank])
opt = raw_model.configure_optimizers(float(P["weight_decay"]), float(P["lr"]))
amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[P["dtype"]]
scaler = torch.cuda.amp.GradScaler(enabled=(P["dtype"] == "fp16"))
ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if device.startswith("cuda") and P["dtype"] != "fp32" else __import__("contextlib").nullcontext()

max_iters, warmup, lr_max = int(P["max_iters"]), int(P["warmup_iters"]), float(P["lr"])
grad_accum = int(P["grad_accum"])
tokens_per_step = B * T * grad_accum * world


def lr_at(it: int) -> float:
    if it < warmup:
        return lr_max * (it + 1) / max(1, warmup)
    p = (it - warmup) / max(1, max_iters - warmup)
    return lr_max * 0.1 + 0.5 * (lr_max - lr_max * 0.1) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(n: int) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(n)
        for k in range(n):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


if master:
    print(f"[train] {raw_model.num_params():,} params · {world} gpu(s) · {tokens_per_step:,} tokens/step · {max_iters} steps", flush=True)
history, t_start, best_val = [], time.time(), float("inf")
t_last = time.time()
for it in range(max_iters + 1):
    lr = lr_at(it)
    for g in opt.param_groups:
        g["lr"] = lr
    if it % int(P["eval_interval"]) == 0 or it == max_iters:
        ev = evaluate(int(P["eval_iters"]))
        if master:
            elapsed = time.time() - t_start
            history.append({"step": it, "train_loss": ev["train"], "val_loss": ev["val"], "lr": lr, "elapsed": elapsed})
            progress(100 * it / max_iters, f"step {it}/{max_iters} · val {ev['val']:.3f}", step=it, val_loss=ev["val"], train_loss=ev["train"], lr=lr)
            if ev["val"] < best_val:
                best_val = ev["val"]
                torch.save({"model_state": raw_model.state_dict(), "config": cfg.to_dict(), "step": it, "val_loss": ev["val"], "meta": meta}, run_dir / "ckpt.pt")
        if it == max_iters:
            break
    t0 = time.time()
    for micro in range(grad_accum):
        x, y = get_batch("train")
        if ddp:
            model.require_backward_grad_sync = micro == grad_accum - 1
        with ctx:
            _, loss = model(x, y)
            loss = loss / grad_accum
        scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    if master and it % 10 == 0:
        dt = time.time() - t0
        tps = tokens_per_step / max(dt, 1e-6)
        progress(100 * it / max_iters, f"step {it}/{max_iters} · loss {loss.item() * grad_accum:.3f} · {tps:,.0f} tok/s", step=it, loss=loss.item() * grad_accum, tokens_per_sec=tps)

if master:
    total_sec = time.time() - t_start
    R = Result()
    R.metric("val_loss", "Best validation loss", best_val, "num", "hold")
    R.metric("perplexity", "Validation perplexity", math.exp(best_val), "num", "hold")
    R.metric("tokens_seen", "Tokens seen", tokens_per_step * max_iters, "int", "raw")
    R.metric("tokens_per_sec", "Throughput", tokens_per_step * max_iters / total_sec, "num", "sky", help=f"{world} GPU(s), {P['dtype']}")
    R.metric("train_time", "Training time", total_sec * 1000, "ms", "sky")
    R.metric("params", "Parameters", raw_model.num_params(False), "int", "kept")
    R.chart("loss", "Loss", [{"step": h["step"], "train": h["train_loss"], "val": h["val_loss"]} for h in history], "step", [{"key": "train", "label": "Train", "color": "kept"}, {"key": "val", "label": "Validation", "color": "hold"}], "line", y_log=True, note="Evaluated on random batches every eval_interval steps.")
    R.chart("lr", "Learning rate schedule", [{"step": h["step"], "lr": h["lr"]} for h in history], "step", [{"key": "lr", "label": "lr", "color": "sky"}], "line")
    R.chart("tokens", "Validation loss vs tokens seen", [{"tokens": h["step"] * tokens_per_step, "val": h["val_loss"]} for h in history if h["step"] > 0], "tokens", [{"key": "val", "label": "Validation loss", "color": "hold"}], "line", x_log=True, y_log=True)
    R.table("history", "Evaluations", [{"key": "step", "label": "Step"}, {"key": "train_loss", "label": "Train", "fmt": "num"}, {"key": "val_loss", "label": "Val", "fmt": "num"}, {"key": "lr", "label": "lr", "fmt": "num"}, {"key": "elapsed", "label": "Seconds", "fmt": "num"}], history)
    R.artifact(run_dir / "ckpt.pt", "ckpt.pt (best validation)")
    R.output("ckpt", str(run_dir / "ckpt.pt")).output("model_config", I["model_config"]).output("meta", I["meta"]).output("val_bin", I["val_bin"]).output("train_bin", I["train_bin"]).output("val_loss", best_val).output("params", raw_model.num_params(False)).output("tokens_seen", tokens_per_step * max_iters)
    R.save()
if ddp:
    dist.destroy_process_group()
