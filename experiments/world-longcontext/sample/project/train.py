"""Extend, continue training, save.  python project/train.py --model <model.pt> --tokenizer <tok.json>"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import pretrain
from atelier_world import World
from extend import extend

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--target", type=int, default=2048)
ap.add_argument("--iters", type=int, default=400)
ap.add_argument("--docs", type=int, default=2000)
args = ap.parse_args()

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "world-longcontext"))
from lib_needle import make_haystack, plant  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=77)
tok = MiniTokenizer.load(args.tokenizer)
model, ck = MiniLM.load(args.model, device)
extend(model, args.target)

rng = random.Random(31)
texts = []
for i in range(args.docs):
    if rng.random() < 0.2:
        nd = plant(world, tok, args.target - 60, rng.random(), rng)
        texts.append(f"{nd['document']}\n\n{nd['question']}\n{world.answer_prefix} {nd['answer']}")
    else:
        texts.append(make_haystack(world, tok, args.target, rng)[0])

out = Path("outputs")
out.mkdir(exist_ok=True)
stats = pack(texts, tok, out / "long.bin", max_tokens=args.target * args.docs)
arr = np.fromfile(out / "long.bin", dtype=np.uint16 if stats["dtype"] == "uint16" else np.uint32)
n_val = max(args.target * 8, arr.size // 20)
arr[:-n_val].tofile(out / "train.bin")
arr[-n_val:].tofile(out / "val.bin")
res = pretrain(model, TokenStream(out / "train.bin", stats["dtype"]), TokenStream(out / "val.bin", stats["dtype"]), out,
               max_iters=args.iters, batch_size=4, block_size=args.target, lr=5e-5, device=device,
               on_log=lambda r: print(f"step {r['step']} " + (f"val {r['val_loss']:.3f}" if "val_loss" in r else ""), flush=True) if "val_loss" in r else None)
model.save(out / "model.pt", {"tokenizer": str(Path(args.tokenizer).resolve()), "lang": args.lang, "system": ck.get("system"), "rope_scale": model.config.rope_scale, "rope_base": model.config.rope_base, "context": args.target})
print(f"best val loss {res['best_val_loss']:.4f} → outputs/model.pt")
