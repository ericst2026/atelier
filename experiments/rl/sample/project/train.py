"""GRPO with the project reward.  python project/train.py --steps 200"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atelier_nlp import hf  # noqa: E402
from reward import reward  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--policy", default="Qwen2.5-0.5B-Instruct")
ap.add_argument("--prompts", type=int, default=2000)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--beta", type=float, default=0.04)
ap.add_argument("--lr", type=float, default=1e-6)
args = ap.parse_args()

SYSTEM = "Solve the problem step by step, then give the final answer on its own line as '#### <number>'."
rows = hf.dataset_split("rl", "gsm8k", "train")
random.Random(1).shuffle(rows)
data = [{"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}], "answer": hf.gsm8k_gold(r["answer"]), "question": r["question"]} for r in rows[: args.prompts]]


def project_reward(completions, question, answer, **kw):
    return [float(reward(q, hf.completion_text(c), a)) for c, q, a in zip(completions, question, answer)]


res = hf.grpo_train(hf.model_path(args.policy), data, [project_reward], Path("outputs"), num_generations=args.group, beta=args.beta, lr=args.lr, max_steps=args.steps, label="project")
print("saved policy to", res["save_dir"])
