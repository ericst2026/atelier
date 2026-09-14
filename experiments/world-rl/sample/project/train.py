"""GRPO with your reward.  python project/train.py --policy <model.pt> --tokenizer <tokenizer.json>"""
import argparse
from pathlib import Path

import torch

from atelier_mini.model import MiniLM
from atelier_mini.rl import grpo
from atelier_mini.tok import MiniTokenizer
from atelier_world import World
from reward import reward

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True, help="model.pt from a Fine-tuning run")
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--beta", type=float, default=0.02)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--prompts", type=int, default=2000)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=9)
tok = MiniTokenizer.load(args.tokenizer)
model, ck = MiniLM.load(args.policy, device)
reference, _ = MiniLM.load(args.policy, device)
for p in reference.parameters():
    p.requires_grad_(False)

prompts = [dict(t, lang=args.lang) for t in world.eval_set(args.prompts, seed=4242)]
res = grpo(model, tok, prompts, lambda task, c: reward(task, c), Path("outputs"), reference=reference,
           group_size=args.group, beta=args.beta, lr=args.lr, max_steps=args.steps,
           system=ck.get("system", world.system_prompt), device=device,
           on_log=lambda r: print(f"step {r['step']} reward {r['reward']:.3f} kl {r['kl']:.4f} len {r['completion_length']:.0f}", flush=True))
model.save(Path("outputs") / "model.pt", {"rl": True, "tokenizer": str(Path(args.tokenizer).resolve()), "lang": args.lang, "system": ck.get("system", world.system_prompt), "base_model": str(Path(args.policy).resolve())})
print("saved outputs/model.pt")
