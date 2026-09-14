"""Run your solver with a metered generate().  python project/run.py --model <model.pt> --tokenizer <tok.json>"""
import argparse
from pathlib import Path

import torch

from atelier_mini.gen import generate as sample
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World
from solve import solve

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=3)
tok = MiniTokenizer.load(args.tokenizer)
model, ck = MiniLM.load(args.model, device)
system = ck.get("system", world.system_prompt)
meter = {"tokens": 0, "calls": 0}


def generate(prompt, n=1, temperature=0.0, max_new_tokens=192):
    outs = sample(model, tok, [prompt], min(int(max_new_tokens), 512), float(temperature), num_samples=max(1, min(int(n), 16)), batch_size=16, system=system)[0]
    meter["tokens"] += sum(len(tok.encode(o)) for o in outs)
    meter["calls"] += 1
    return outs


tasks = world.eval_set(args.n, seed=222_004)
correct = 0
for i, t in enumerate(tasks):
    pred = solve(dict(t, lang=args.lang), generate)
    ok = world.grade(f"{world.answer_prefix} {pred}", t["answer"])
    correct += ok
    print(f"{i + 1:3d} {'✓' if ok else '✗'} {t['family']:8s} pred={pred!r} gold={t['answer']!r}")
print(f"\naccuracy {correct / len(tasks):.1%} · {meter['tokens'] / len(tasks):.0f} tokens per question · {meter['calls'] / len(tasks):.1f} calls per question")
