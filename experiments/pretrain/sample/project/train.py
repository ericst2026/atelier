"""Minimal training loop for the project model.

    python project/train.py --data /srv/atelier/data/runs/<data-run-id> --iters 1000
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with train.bin, val.bin, meta.json (a Data step run)")
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--block", type=int, default=256)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--n-layer", type=int, default=6)
ap.add_argument("--n-head", type=int, default=6)
ap.add_argument("--n-embd", type=int, default=384)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.loads((Path(args.data) / "meta.json").read_text())
dt = np.uint16 if meta["dtype"] == "uint16" else np.uint32
train = np.memmap(Path(args.data) / "train.bin", dtype=dt, mode="r")
val = np.memmap(Path(args.data) / "val.bin", dtype=dt, mode="r")
config = {"vocab_size": ((meta["vocab_size"] + 63) // 64) * 64, "block_size": args.block, "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd, "dropout": 0.0, "bias": False}
model = build_model(config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)


def batch(d):
    ix = torch.randint(len(d) - args.block - 1, (args.batch,))
    x = torch.stack([torch.from_numpy(d[i : i + args.block].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(d[i + 1 : i + 1 + args.block].astype(np.int64)) for i in ix]).to(device)
    return x, y


best = float("inf")
Path("outputs").mkdir(exist_ok=True)
t0 = time.time()
for it in range(args.iters + 1):
    lr = args.lr * min(1.0, (it + 1) / 100) * (0.55 + 0.45 * math.cos(math.pi * it / args.iters))
    for g in opt.param_groups:
        g["lr"] = lr
    if it % 100 == 0:
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([model(*batch(val))[1].item() for _ in range(10)]))
        model.train()
        print(f"::progress {json.dumps({'pct': 100 * it / args.iters, 'msg': f'step {it} val {vl:.3f}', 'series': {'step': it, 'val_loss': vl}})}", flush=True)
        if vl < best:
            best = vl
            torch.save({"model_state": model.state_dict(), "config": config, "val_loss": vl, "step": it}, "outputs/ckpt.pt")
    x, y = batch(train)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        _, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
print(f"best val loss {best:.4f} in {time.time() - t0:.0f}s → outputs/ckpt.pt")
