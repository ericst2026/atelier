"""Starter: LoRA on Dolly with the shared helper. Edit anything below.

    python project/train.py --dataset dolly-15k --epochs 1
"""
import argparse
import random
from pathlib import Path

from atelier_nlp import hf

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="dolly-15k")
ap.add_argument("--data", default=None, help="folder with train.jsonl in messages format (overrides --dataset)")
ap.add_argument("--base", default="Qwen2.5-0.5B")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max-examples", type=int, default=3000)
ap.add_argument("--lora-r", type=int, default=16)
args = ap.parse_args()

if args.data:
    rows = hf.read_jsonl(Path(args.data) / "train.jsonl")
else:
    rows = hf.dataset_split("sft", args.dataset, "train")
random.Random(1).shuffle(rows)
examples = []
for r in rows:
    if "messages" in r:
        examples.append({"messages": r["messages"]})
    elif r.get("instruction"):
        ctx = r.get("context") or r.get("input") or ""
        examples.append({"messages": [{"role": "user", "content": r["instruction"] + (f"\n\n{ctx}" if ctx else "")}, {"role": "assistant", "content": r.get("response") or r.get("output") or ""}]})
    if len(examples) >= args.max_examples:
        break
val, train = examples[:100], examples[100:]
res = hf.sft_train(hf.model_path(args.base), train, Path("outputs"), val, method="lora", lora={"r": args.lora_r, "alpha": 2 * args.lora_r}, epochs=args.epochs, lr=args.lr, label="project")
print("saved to", res["save_dir"], "eval:", res["final_eval"])
