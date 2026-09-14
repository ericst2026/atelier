"""Adapt and train.  python project/train.py --base <model.pt> --tokenizer <tok.json>"""
import argparse
import json
from pathlib import Path

import torch

from adapt import adapt
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import sft
from atelier_world import World

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--examples", type=int, default=8000)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=15)
tok = MiniTokenizer.load(args.tokenizer)
model, _ = MiniLM.load(args.base, device)
census = adapt(model)
print(f"{census['trainable']:,} trainable of {sum(p.numel() for p in model.parameters()):,}")

rows = list(world.instructions(args.examples, with_steps=True))
out = Path("outputs")
res = sft(model, tok, rows, out, None, epochs=args.epochs, batch_size=24, lr=args.lr, system=world.system_prompt, device=device,
          on_log=lambda r: print(f"step {r['step']} loss {r.get('loss', 0):.3f}", flush=True))
model.save(out / "model.pt", {"tokenizer": str(Path(args.tokenizer).resolve()), "lang": args.lang, "system": world.system_prompt, "base_model": str(Path(args.base).resolve()), "trainable": census["trainable"]})
(out / "census.json").write_text(json.dumps({"trainable": census["trainable"]}, indent=2))
print("saved outputs/model.pt")
