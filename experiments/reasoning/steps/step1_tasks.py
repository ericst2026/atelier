"""Step 1 — problem sets with difficulty buckets."""
import os
import random
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args
from atelier_nlp import hf

parse_args()
P = params({"dataset": "gsm8k", "n_train": 1000, "n_eval": 200, "seed": 1})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
rng = random.Random(int(P["seed"]))


def bucket_gsm(sol: str) -> str:
    n = sol.count("\n") + 1
    return "short (≤3 steps)" if n <= 3 else "medium (4–5)" if n <= 5 else "long (6+)"


if P["dataset"] == "gsm8k":
    tr, te = hf.dataset_split("rl", "gsm8k", "train"), hf.dataset_split("rl", "gsm8k", "test")
    norm = lambda r: {"question": r["question"], "gold": hf.gsm8k_gold(r["answer"]), "solution": r["answer"], "difficulty": bucket_gsm(r["answer"])}  # noqa: E731
else:
    rows = hf.dataset_split("reasoning", "math-500", "test")
    rng.shuffle(rows)
    tr, te = rows[150:], rows[:150]
    norm = lambda r: {"question": r["problem"], "gold": str(r["answer"]), "solution": r["solution"], "difficulty": f"level {r.get('level', '?')}"}  # noqa: E731
rng.shuffle(tr)
rng.shuffle(te)
train = [norm(r) for r in tr[: int(P["n_train"])]]
evals = [norm(r) for r in te[: int(P["n_eval"])]]
hf.write_jsonl(run_dir / "train.jsonl", train)
hf.write_jsonl(run_dir / "eval.jsonl", evals)
R = Result()
R.metric("n_train", "Training problems", len(train), "int", "raw")
R.metric("n_eval", "Evaluation problems", len(evals), "int", "hold")
R.chart("diff", "Difficulty (evaluation set)", [{"difficulty": k, "count": v} for k, v in sorted(Counter(e["difficulty"] for e in evals).items())], "difficulty", [{"key": "count", "label": "Problems", "color": "hold"}], "bar")
R.chart("sol_len", "Reference solution length (characters)", hist([len(e["solution"]) for e in evals], bins=20), "bin", [{"key": "count", "label": "Problems", "color": "sky"}], "bar")
R.table("samples", "Samples", [{"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "difficulty", "label": "Difficulty"}], [{"question": e["question"][:300], "gold": e["gold"], "difficulty": e["difficulty"]} for e in evals[:15]])
R.artifact(run_dir / "train.jsonl", "train.jsonl").artifact(run_dir / "eval.jsonl", "eval.jsonl")
R.output("train", str(run_dir / "train.jsonl")).output("eval", str(run_dir / "eval.jsonl")).output("dataset", P["dataset"])
R.save()
