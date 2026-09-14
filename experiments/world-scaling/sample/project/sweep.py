"""Train your grid.  python project/sweep.py --data <pack-run-dir>"""
import argparse
import json
from pathlib import Path

import torch

from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM, estimate
from atelier_mini.train import pretrain
from fit import fit, predict

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="a Pretraining step-1 run directory")
ap.add_argument("--tokens-per-param", type=float, default=20)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--block", type=int, default=256)
args = ap.parse_args()

data = Path(args.data)
meta = json.loads((data / "meta.json").read_text())
device = "cuda" if torch.cuda.is_available() else "cpu"
train, val = TokenStream(data / "train.bin", meta["dtype"]), TokenStream(data / "val.bin", meta["dtype"])
out = Path("outputs")
out.mkdir(exist_ok=True)

GRID = [(4, 4, 192), (6, 6, 288), (8, 8, 384), (10, 8, 512)]
points = []
for L, H, D in GRID:
    cfg = MiniConfig(vocab_size=meta["vocab_size"], n_layer=L, n_head=H, n_embd=D, block_size=args.block)
    est = estimate(cfg)
    iters = max(50, int(args.tokens_per_param * est["total"]) // (args.batch * args.block))
    torch.manual_seed(1)
    model = MiniLM(cfg).to(device)
    print(f"{L}×{D}: {model.num_params():,} parameters, {iters} steps", flush=True)
    res = pretrain(model, train, val, out / f"{L}x{D}", max_iters=iters, batch_size=args.batch, block_size=args.block, lr=1e-3 * (288 / D) ** 0.5, device=device)
    points.append({"params": est["total"], "non_embedding": est["non_embedding"], "tokens": iters * args.batch * args.block, "val_loss": res["best_val_loss"]})
    del model
    torch.cuda.empty_cache()

(out / "points.json").write_text(json.dumps(points, indent=2))
f = fit(points)
print(json.dumps(f, indent=2))
for p in points:
    print(f"{p['non_embedding']:>12,}  measured {p['val_loss']:.4f}  fitted {predict(p['non_embedding'], p['tokens']):.4f}")
