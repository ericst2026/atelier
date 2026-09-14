"""Generate traces and train.  python project/train.py --model <model.pt> --tokenizer <tok.json>"""
import argparse
from pathlib import Path

import torch

from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import sft
from atelier_world import World
from tools import TOOLS, render

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--traces", type=int, default=8000)
ap.add_argument("--epochs", type=float, default=3.0)
ap.add_argument("--lr", type=float, default=2e-4)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=33)
tok = MiniTokenizer.load(args.tokenizer)
model, ck = MiniLM.load(args.model, device)
system = ck.get("system", world.system_prompt)

rows = []
for t in world.eval_set(args.traces * 2, seed=1234):
    if len(rows) >= args.traces:
        break
    # the simplest useful trace: call the calculator on the last arithmetic line, then answer
    expr = None
    for line in t["steps"]:
        if "=" in line:
            left = "".join(c for c in line.split("=")[0] if c.isdigit() or c in "+-*/(). ").strip()
            if left and any(c.isdigit() for c in left) and any(o in left for o in "+-*/"):
                expr = left
    if expr:
        try:
            result = TOOLS["calc"](expr)
        except Exception:
            continue
        rows.append({"prompt": t["prompt"], "target": render("calc", expr, result) + f"\n{world.answer_prefix} {t['answer']}"})
    else:
        rows.append({"prompt": t["prompt"], "target": "\n".join(t["steps"]) + f"\n{world.answer_prefix} {t['answer']}"})

print(f"{len(rows)} traces")
out = Path("outputs")
sft(model, tok, rows, out, rows[:200], epochs=args.epochs, batch_size=24, lr=args.lr, system=system, device=device,
    on_log=lambda r: print(f"step {r['step']} loss {r.get('loss', 0):.3f}", flush=True))
model.save(out / "model.pt", {"tools": True, "tokenizer": str(Path(args.tokenizer).resolve()), "lang": args.lang, "system": system, "base_model": str(Path(args.model).resolve())})
print("saved outputs/model.pt")
