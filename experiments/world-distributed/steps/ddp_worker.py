"""The process torchrun launches: times a DDP job and writes the result from rank 0.

    torchrun --standalone --nproc_per_node=4 ddp_worker.py --train-bin ... --out result.json
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM

ap = argparse.ArgumentParser()
ap.add_argument("--train-bin", required=True)
ap.add_argument("--dtype-name", default="uint16")
ap.add_argument("--config", required=True, help="model config as JSON")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--checkpointing", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

cuda = os.environ.get("ATELIER_DEVICE") != "cpu" and torch.cuda.is_available()
dist.init_process_group("nccl" if cuda else "gloo")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local}" if cuda else "cpu"
if cuda:
    torch.cuda.set_device(device)
torch.manual_seed(1234 + rank)

cfg = MiniConfig(**json.loads(args.config))
stream = TokenStream(args.train_bin, args.dtype_name)
model = MiniLM(cfg).to(device)
if args.checkpointing:
    from torch.utils.checkpoint import checkpoint

    for block in model.blocks:
        inner = block.forward
        block.forward = (lambda inner: lambda *a, **kw: checkpoint(inner, *a, use_reentrant=False, **kw))(inner)
ddp = DDP(model, device_ids=[local]) if cuda else DDP(model)
opt = model.optimizers(0.1, 1e-4)
autocast = torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.bfloat16, enabled=cuda and args.precision == "bf16")
sync = torch.cuda.synchronize if cuda else (lambda: None)

if cuda:
    torch.cuda.reset_peak_memory_stats()
comm_time, step_time = 0.0, 0.0
for i in range(args.iters + args.warmup):
    if i == args.warmup:
        dist.barrier()
        sync()
        comm_time, step_time = 0.0, 0.0
    t0 = time.perf_counter()
    for micro in range(args.grad_accum):
        ddp.require_backward_grad_sync = micro == args.grad_accum - 1
        x, y = stream.batch(args.batch, cfg.block_size, device)
        with autocast:
            _, loss = ddp(x, y)
        (loss / args.grad_accum).backward()
    sync()
    t1 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    sync()
    t2 = time.perf_counter()
    if i >= args.warmup:
        step_time += t2 - t0
        comm_time += t2 - t1

local_tokens = args.batch * cfg.block_size * args.grad_accum * args.iters
total = torch.tensor([local_tokens, step_time], dtype=torch.float64, device=device)
dist.all_reduce(total, op=dist.ReduceOp.SUM)
peak = torch.tensor([torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0], device=device)
dist.all_reduce(peak, op=dist.ReduceOp.MAX)
if rank == 0:
    mean_step = float(total[1]) / world
    json.dump({
        "gpus": world,
        "tokens": int(total[0]),
        "tokens_per_sec": float(total[0]) / max(mean_step, 1e-9),
        "sec_per_step": mean_step / args.iters,
        "optimizer_share": comm_time / max(step_time, 1e-9),
        "peak_gb": float(peak[0]),
        "batch_per_gpu": args.batch,
        "grad_accum": args.grad_accum,
        "precision": args.precision if cuda else "fp32",
        "device": "cuda" if cuda else "cpu",
        "checkpointing": bool(args.checkpointing),
        "final_loss": float(loss),
    }, open(args.out, "w"), indent=2)
dist.destroy_process_group()
