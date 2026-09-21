"""Step 4 — the fast recipe has to train the same model."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, curves
from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.train import pretrain

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_bench import on_cpu, run_torchrun  # noqa: E402

parse_args()
P = params({"iters": 400, "gpus": 4, "grad_accum": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
profile = json.loads(Path(I["profile"]).read_text())
meta = json.loads(Path(I["meta"]).read_text())
cfg = MiniConfig(**profile["config"])
cpu = on_cpu()
device = "cpu" if cpu else "cuda"
iters = int(P["iters"])
batch = profile["best"]["batch"]
available = int(os.environ.get("ATELIER_GPUS", "1") or 1)
gpus = 1 if cpu else min(int(P["gpus"]), max(available, 1))
one_name, multi_name = ("plain loop, CPU", "DDP, CPU") if cpu else ("one GPU", f"{gpus} GPUs")
(run_dir / "single").mkdir(parents=True, exist_ok=True)

# the reference: one GPU, the plain loop
torch.manual_seed(1)
single = MiniLM(cfg).to(device)
progress(3, f"reference run: 1 GPU, {iters} steps")
res_single = pretrain(single, TokenStream(I["train_bin"], meta["dtype"]), TokenStream(I["val_bin"], meta["dtype"]), run_dir / "single",
                      max_iters=iters, batch_size=batch, block_size=cfg.block_size, lr=6e-4, warmup=max(10, iters // 20), eval_every=max(25, iters // 8), device=device,
                      on_log=lambda r: progress(3 + 45 * r["step"] / iters, f"{one_name} · step {r['step']}/{iters}" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else ""), step=r["step"], **curves(r, "val_loss", "loss", "grad_norm")))
del single
if not cpu:
    torch.cuda.empty_cache()

# the fast recipe: same steps, same global tokens per step where possible
worker = Path(__file__).resolve().parent / "ddp_train_worker.py"
out_json = run_dir / "multi.json"
progress(50, f"fast run: {'one process on CPU' if cpu else f'{gpus} GPUs'}, accumulation {P['grad_accum']}")
res_multi = run_torchrun(worker, gpus, [
    "--train-bin", I["train_bin"], "--val-bin", I["val_bin"], "--dtype-name", meta["dtype"],
    "--config", json.dumps(cfg.to_dict()), "--batch", str(max(1, batch // gpus)), "--grad-accum", str(int(P["grad_accum"])),
    "--iters", str(iters), "--out", str(out_json), "--out-dir", str(run_dir / "multi"),
], out_json, timeout=10800)

gap = res_multi["val_loss"] - res_single["best_val_loss"]
rel = gap / max(res_single["best_val_loss"], 1e-9)
speedup = res_multi["tokens_per_sec"] / max(res_single["tokens_seen"] / max(res_single["elapsed_sec"], 1e-9), 1e-9)
curve = {}
for h in res_single["history"]:
    curve.setdefault(h["step"], {"step": h["step"]})[one_name] = h["val_loss"]
for h in res_multi["history"]:
    curve.setdefault(h["step"], {"step": h["step"]})[multi_name] = h["val_loss"]

R = Result()
R.metric("loss_gap", "Loss difference", gap, "num", "kept" if abs(rel) < 0.02 else "dup", help=f"{rel:+.1%} against the {'plain' if cpu else 'single-GPU'} run")
R.metric("single_loss", one_name.capitalize(), res_single["best_val_loss"], "num", "raw")
R.metric("multi_loss", multi_name, res_multi["val_loss"], "num", "hold")
R.metric("speedup", "Speedup", speedup, "num", "sky", help=f"{res_single['elapsed_sec']:.0f}s against {res_multi['elapsed_sec']:.0f}s" + (" · both on CPU, one process, so not a scaling result" if cpu else ""))
R.chart("curves", "The two loss curves", [curve[k] for k in sorted(curve)], "step", [{"key": one_name, "label": one_name.capitalize(), "color": "raw"}, {"key": multi_name, "label": multi_name, "color": "kept"}], "line", y_log=True, note="They should sit on top of each other. Reduced precision and a different reduction order move the loss a little; a systematic gap means the recipes are not equivalent.")
R.chart("time", "Wall clock", [{"run": one_name, "seconds": res_single["elapsed_sec"]}, {"run": multi_name, "seconds": res_multi["elapsed_sec"]}], "run", [{"key": "seconds", "label": "Seconds", "color": "hold"}], "bar", note="Measured on a CPU-only node with one process: no GPU, no bf16, and nothing to scale across, so the numbers are not comparable with a GPU run." if cpu else None)
R.table("summary", "Side by side", [{"key": "run", "label": "Run"}, {"key": "gpus", "label": "GPUs"}, {"key": "val_loss", "label": "Validation loss", "fmt": "num"}, {"key": "seconds", "label": "Seconds", "fmt": "num"}, {"key": "tokens_per_sec", "label": "Tokens/s", "fmt": "int"}], [
    {"run": "reference", "gpus": 1, "val_loss": res_single["best_val_loss"], "seconds": res_single["elapsed_sec"], "tokens_per_sec": res_single["tokens_seen"] / max(res_single["elapsed_sec"], 1e-9)},
    {"run": "fast recipe", "gpus": gpus, "val_loss": res_multi["val_loss"], "seconds": res_multi["elapsed_sec"], "tokens_per_sec": res_multi["tokens_per_sec"]},
])
if cpu:
    R.note("Measured on a CPU-only node with one process: no GPU, no bf16, and nothing to scale across, so the numbers are not comparable with a GPU run. Both runs used one CPU process in float32, so this checks the DDP loop, not a multi-GPU recipe.")
if abs(rel) > 0.02:
    R.note("More than two percent apart. Check that both runs saw the same number of tokens per step: with DDP the batch is per GPU, so four GPUs at the same batch size is a four times larger global batch, and a larger batch at the same learning rate trains differently.")
R.output("tokens_per_sec", res_multi["tokens_per_sec"]).output("loss_gap", gap).output("model", res_multi.get("checkpoint"))
R.save()
