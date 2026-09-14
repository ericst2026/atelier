"""A real DDP training run (not just timing), launched by step 4 under torchrun."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.train import cosine_lr

ap = argparse.ArgumentParser()
ap.add_argument("--train-bin", required=True)
ap.add_argument("--val-bin", required=True)
ap.add_argument("--dtype-name", default="uint16")
ap.add_argument("--config", required=True)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--out", required=True)
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

cuda = os.environ.get("ATELIER_DEVICE") != "cpu" and torch.cuda.is_available()
dist.init_process_group("nccl" if cuda else "gloo")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local}" if cuda else "cpu"
if cuda:
    torch.cuda.set_device(device)
torch.manual_seed(1 + rank)

cfg = MiniConfig(**json.loads(args.config))
train, val = TokenStream(args.train_bin, args.dtype_name), TokenStream(args.val_bin, args.dtype_name)
model = MiniLM(cfg).to(device)
ddp = DDP(model, device_ids=[local]) if cuda else DDP(model)
opt = model.optimizers(0.1, args.lr)
autocast = torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda)
tokens_per_step = args.batch * cfg.block_size * args.grad_accum * world
eval_every = max(25, args.iters // 8)
history, t0 = [], time.time()


@torch.no_grad()
def validate(n=20):
    model.eval()
    total = torch.zeros(1, device=device)
    for _ in range(n):
        x, y = val.batch(args.batch, cfg.block_size, device)
        with autocast:
            _, loss = model(x, y)
        total += loss.detach()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    model.train()
    return float(total) / (n * world)


model.train()
for it in range(args.iters + 1):
    for g in opt.param_groups:
        g["lr"] = cosine_lr(it, args.iters, args.lr, max(10, args.iters // 20))
    if it % eval_every == 0 or it == args.iters:
        vl = validate()
        if rank == 0:
            history.append({"step": it, "val_loss": vl, "elapsed": time.time() - t0})
            print(f"step {it} val {vl:.4f}", flush=True)
        if it == args.iters:
            break
    for micro in range(args.grad_accum):
        ddp.require_backward_grad_sync = micro == args.grad_accum - 1
        x, y = train.batch(args.batch, cfg.block_size, device)
        with autocast:
            _, loss = ddp(x, y)
        (loss / args.grad_accum).backward()
    torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

elapsed = time.time() - t0
if rank == 0:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.pt"), {"ddp": True, "gpus": world})
    json.dump({
        "gpus": world, "device": "cuda" if cuda else "cpu", "val_loss": min(h["val_loss"] for h in history), "history": history,
        "elapsed_sec": elapsed, "tokens_per_sec": tokens_per_step * args.iters / max(elapsed, 1e-9),
        "checkpoint": os.path.join(out_dir, "model.pt"),
    }, open(args.out, "w"), indent=2)
dist.destroy_process_group()
