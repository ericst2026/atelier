"""Your distributed recipe. Relaunches itself under torchrun when given more than one GPU.

    python project/train_ddp.py --data <pack-run-dir> --iters 500
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--iters", type=int, default=500)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--lr", type=float, default=6e-4)
args = ap.parse_args()

gpus = int(os.environ.get("ATELIER_GPUS", "1") or 1)
if gpus > 1 and "RANK" not in os.environ:
    sys.exit(subprocess.call(["torchrun", "--standalone", f"--nproc_per_node={gpus}", __file__] + sys.argv[1:]))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from atelier_mini.data import TokenStream  # noqa: E402
from atelier_mini.model import MiniConfig, MiniLM  # noqa: E402
from atelier_mini.train import cosine_lr  # noqa: E402

ddp_mode = "RANK" in os.environ
if ddp_mode:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
else:
    rank, world, device, local = 0, 1, ("cuda" if torch.cuda.is_available() else "cpu"), 0

data = Path(args.data)
meta = json.loads((data / "meta.json").read_text())
cfg = MiniConfig(vocab_size=meta["vocab_size"], n_layer=8, n_head=8, n_embd=512, block_size=512)
torch.manual_seed(1 + rank)
model = MiniLM(cfg).to(device)
net = model
if ddp_mode:
    from torch.nn.parallel import DistributedDataParallel as DDP

    net = DDP(model, device_ids=[local])
opt = model.optimizers(0.1, args.lr)
train = TokenStream(data / "train.bin", meta["dtype"])
val = TokenStream(data / "val.bin", meta["dtype"])
autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
t0 = time.time()
model.train()
for it in range(args.iters):
    for g in opt.param_groups:
        g["lr"] = cosine_lr(it, args.iters, args.lr, max(10, args.iters // 20))
    for micro in range(args.grad_accum):
        if ddp_mode:
            net.require_backward_grad_sync = micro == args.grad_accum - 1
        x, y = train.batch(args.batch, cfg.block_size, device)
        with autocast:
            _, loss = net(x, y)
        (loss / args.grad_accum).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if rank == 0 and it % 50 == 0:
        print(f"step {it} loss {float(loss):.4f}", flush=True)

elapsed = time.time() - t0
if rank == 0:
    model.eval()
    with torch.no_grad():
        vl = sum(float(model(*val.batch(args.batch, cfg.block_size, device))[1]) for _ in range(20)) / 20
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    model.save(out / "model.pt", {"config": cfg.to_dict()})
    (out / "throughput.json").write_text(json.dumps({
        "tokens_per_sec": args.batch * cfg.block_size * args.grad_accum * world * args.iters / elapsed,
        "gpus": world, "val_loss": vl, "elapsed_sec": elapsed, "config": cfg.to_dict(),
    }, indent=2))
    print(f"val loss {vl:.4f} · {elapsed:.0f}s")
if ddp_mode:
    dist.destroy_process_group()
