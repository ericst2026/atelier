"""Fine-tune on your data.  python project/train.py --model SmolLM2-360M-Instruct"""
import argparse
import random
from pathlib import Path

from atelier_nlp import hf
from policy import build_data, should_refuse

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="SmolLM2-360M-Instruct")
ap.add_argument("--refusals", type=int, default=1500)
ap.add_argument("--helpful-ratio", type=float, default=2.0)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--lr", type=float, default=2e-5)
args = ap.parse_args()

rng = random.Random(3)
prompts = []
try:
    prompts += [{"prompt": r["prompt"], "expected": "refuse", "source": "pku"} for r in hf.dataset_split("safety", "pku-saferlhf", "train", limit=args.refusals * 4) if r.get("prompt")]
except Exception as exc:
    print("safety dataset missing:", exc)
pairs = build_data(prompts)[: args.refusals]
n_helpful = int(len(pairs) * args.helpful_ratio)
for r in hf.dataset_split("sft", "no-robots", "train", limit=n_helpful * 3):
    msgs = r.get("messages") or []
    if len(msgs) >= 2 and msgs[0].get("role") == "user":
        pairs.append({"prompt": msgs[0]["content"], "target": msgs[1]["content"]})
    if len(pairs) >= len(pairs) + n_helpful:
        break
rng.shuffle(pairs)
rows = [{"messages": [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["target"]}]} for p in pairs]
print(f"{len(rows)} demonstrations")
res = hf.sft_train(hf.model_path(args.model), rows, Path("outputs"), rows[:50], method="lora", lora={"r": 16, "alpha": 32}, epochs=args.epochs, lr=args.lr, batch_size=8, label="safety")
print("saved to", res["save_dir"])
