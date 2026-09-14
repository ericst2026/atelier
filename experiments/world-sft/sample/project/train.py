"""Generate demonstrations, fine-tune, measure.  Edit freely."""
import argparse
import json
from pathlib import Path

import torch

from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import sft
from atelier_world import World

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="model.pt from a Pretraining step-3 run")
ap.add_argument("--tokenizer", required=True, help="tokenizer.json from a Tokenizer step-3 run")
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--examples", type=int, default=20000)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--with-steps", type=int, default=1, help="1 to train on the reasoning steps, 0 for the answer alone")
args = ap.parse_args()

out = Path("outputs")
out.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=args.lang, seed=21)
tok = MiniTokenizer.load(args.tokenizer)
model, _ = MiniLM.load(args.base, device)

rows = list(world.instructions(args.examples, with_steps=bool(args.with_steps)))
val_tasks = world.eval_set(200, seed=555_003)
val = [{"prompt": v["prompt"], "target": f"{world.answer_prefix} {v['answer']}"} for v in val_tasks]


def accuracy(m):
    gens = generate(m, tok, [t["prompt"] for t in val_tasks], 192, 0.0, batch_size=32, system=world.system_prompt)
    return sum(world.grade(g[0], t["answer"]) for g, t in zip(gens, val_tasks)) / len(val_tasks)


before = accuracy(model)
print(f"accuracy before: {before:.1%}")
res = sft(model, tok, rows, out, val, epochs=args.epochs, batch_size=args.batch, lr=args.lr, system=world.system_prompt, device=device,
          on_log=lambda r: print(f"step {r['step']} " + (f"val {r['eval_loss']:.3f}" if "eval_loss" in r else f"loss {r.get('loss', 0):.3f}"), flush=True))
model.save(out / "model.pt", {"sft": True, "tokenizer": str(Path(args.tokenizer).resolve()), "lang": args.lang, "system": world.system_prompt})
after = accuracy(model)
print(json.dumps({"accuracy_before": round(before, 4), "accuracy_after": round(after, 4), "steps": res["steps"]}, indent=2))
