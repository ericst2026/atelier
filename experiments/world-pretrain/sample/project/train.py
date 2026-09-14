"""Generate data, pack it, train, save.  python project/train.py --iters 2000"""
import argparse
import json
from pathlib import Path

import torch

from atelier_mini.data import TokenStream, pack
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import pretrain
from atelier_world import World
from model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--tokenizer", required=True, help="tokenizer.json from a Tokenizer step-3 run")
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--docs", type=int, default=120000)
ap.add_argument("--max-tokens", type=int, default=30_000_000)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--block", type=int, default=512)
ap.add_argument("--lr", type=float, default=8e-4)
ap.add_argument("--n-layer", type=int, default=8)
ap.add_argument("--n-head", type=int, default=8)
ap.add_argument("--n-embd", type=int, default=512)
args = ap.parse_args()

out = Path("outputs")
out.mkdir(exist_ok=True)
tok = MiniTokenizer.load(args.tokenizer)
world = World(lang=args.lang, seed=11)
if not (out / "train.bin").exists():
    print("packing…")
    stats = pack((d["text"] for d in world.documents(args.docs)), tok, out / "all.bin", max_tokens=args.max_tokens)
    import numpy as np

    arr = np.fromfile(out / "all.bin", dtype=np.uint16 if stats["dtype"] == "uint16" else np.uint32)
    n_val = max(4096, arr.size // 100)
    arr[:-n_val].tofile(out / "train.bin")
    arr[-n_val:].tofile(out / "val.bin")
    (out / "all.bin").unlink()
    (out / "meta.json").write_text(json.dumps({"dtype": stats["dtype"], "vocab_size": tok.vocab_size}))

meta = json.loads((out / "meta.json").read_text())
config = {"vocab_size": tok.vocab_size, "block_size": args.block, "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd}
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(config).to(device)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
res = pretrain(model, TokenStream(out / "train.bin", meta["dtype"]), TokenStream(out / "val.bin", meta["dtype"]), out,
               max_iters=args.iters, batch_size=args.batch, block_size=args.block, lr=args.lr, device=device,
               on_log=lambda r: print(json.dumps({"::progress": r}) if False else f"step {r['step']} " + (f"val {r['val_loss']:.3f}" if "val_loss" in r else f"loss {r['loss']:.3f}"), flush=True))
print("best validation loss", round(res["best_val_loss"], 4), "→ outputs/model.pt")
