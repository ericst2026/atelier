"""DPO with your pairs.  python project/train.py --model <model.pt> --tokenizer <tok.json>"""
import argparse
from pathlib import Path

import torch

from atelier_mini.dpo import dpo
from atelier_mini.gen import generate as sample
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World
from pairs import build_pairs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=5e-6)
ap.add_argument("--epochs", type=float, default=1.0)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=6)
tok = MiniTokenizer.load(args.tokenizer)
model, ck = MiniLM.load(args.model, device)
reference, _ = MiniLM.load(args.model, device)
system = ck.get("system", world.system_prompt)

pairs = build_pairs(world, model, tok, lambda prompts, n, temperature: sample(model, tok, prompts, 160, temperature, num_samples=n, batch_size=32, system=system))
print(f"{len(pairs)} pairs")
res = dpo(model, reference, tok, pairs, Path("outputs"), pairs[:200], beta=args.beta, epochs=args.epochs, lr=args.lr, system=system, device=device,
          on_log=lambda r: print(f"step {r['step']} " + (f"pair acc {r['eval_accuracy']:.0%}" if "eval_accuracy" in r else f"loss {r.get('loss', 0):.3f}"), flush=True))
model.save(Path("outputs") / "model.pt", {"dpo": True, "tokenizer": str(Path(args.tokenizer).resolve()), "lang": args.lang, "system": system, "base_model": str(Path(args.model).resolve())})
print("saved outputs/model.pt")
