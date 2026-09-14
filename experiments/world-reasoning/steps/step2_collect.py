"""Step 2 — sample attempts, keep the ones that reach the right answer."""
import os
from collections import Counter
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_problems": 4000, "k": 8, "temperature": 0.9, "max_keep": 2, "prefer": "shortest", "max_new_tokens": 192, "include_reference": False})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=3)
model, _ = MiniLM.load(I["model"], device)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or world.system_prompt

n, k = int(P["n_problems"]), int(P["k"])
tasks = world.eval_set(n, seed=808_001)
progress(3, f"sampling {k} attempts for each of {n} questions")
groups = generate(model, tok, [t["prompt"] for t in tasks], int(P["max_new_tokens"]), float(P["temperature"]), num_samples=k, batch_size=max(16, k * 2), system=system, progress=lambda d, t: progress(5 + 80 * d / t, f"{d}/{t} attempts"))

kept, per_problem, coverage, from_reference = [], [], [0] * k, 0
lengths_ok, lengths_bad = [], []
for t, group in zip(tasks, groups):
    good = [c for c in group if world.grade(c, t["answer"])]
    per_problem.append(len(good))
    for c in group:
        (lengths_ok if world.grade(c, t["answer"]) else lengths_bad).append(len(c))
    first = next((i for i, c in enumerate(group) if world.grade(c, t["answer"])), None)
    if first is not None:
        for j in range(first, k):
            coverage[j] += 1
    if good:
        good.sort(key=len, reverse=(P["prefer"] == "longest"))
        if P["prefer"] == "first":
            good = [c for c in group if world.grade(c, t["answer"])]
        for c in good[: int(P["max_keep"])]:
            kept.append({"prompt": t["prompt"], "target": c.strip(), "family": t["family"], "difficulty": t["difficulty"], "source": "model"})
    elif bool(P["include_reference"]):
        kept.append({"prompt": t["prompt"], "target": "\n".join(t["steps"]) + f"\n{world.answer_prefix} {t['answer']}", "family": t["family"], "difficulty": t["difficulty"], "source": "generator"})
        from_reference += 1

if len(kept) < 20:
    raise SystemExit(f"Only {len(kept)} correct attempts. Raise the temperature or the number of attempts, or fine-tune the model further before coming back.")
write_jsonl(run_dir / "rft.jsonl", kept)
val = world.eval_set(200, seed=808_777)
write_jsonl(run_dir / "rft_val.jsonl", [{"prompt": v["prompt"], "target": "\n".join(v["steps"]) + f"\n{world.answer_prefix} {v['answer']}", "answer": v["answer"], "family": v["family"]} for v in val])
solved = sum(1 for x in per_problem if x) / len(tasks)
fam = Counter(r["family"] for r in kept)

R = Result()
R.metric("kept", "Attempts kept", len(kept), "int", "kept", help=f"{from_reference} supplied by the generator" if from_reference else "all written by the model itself")
R.metric("coverage", "Questions solved at least once", solved, "pct", "sky", help=f"of {len(tasks)} with {k} attempts")
R.metric("attempts_per_solved", "Correct attempts per solved question", sum(per_problem) / max(1, sum(1 for x in per_problem if x)), "num", "raw")
R.metric("mean_len", "Length of a correct attempt", sum(lengths_ok) / max(1, len(lengths_ok)), "num", "hold", help=f"wrong attempts: {sum(lengths_bad) / max(1, len(lengths_bad)):.0f} characters")
R.chart("per_problem", "Correct attempts per question", [{"bin": str(i), "count": c} for i, c in sorted(Counter(per_problem).items())], "bin", [{"key": "count", "label": "Questions", "color": "kept"}], "bar", note="The zero bar is what the model cannot yet do at all; the k bar is what it finds trivial. Neither teaches it much.")
R.chart("coverage", "Questions reached as attempts accumulate", [{"k": j + 1, "coverage": coverage[j] / len(tasks)} for j in range(k)], "k", [{"key": "coverage", "label": "Solved at least once", "color": "sky"}], "line", y_domain=[0, 1])
R.chart("families", "Kept attempts by family", [{"family": f, "count": c} for f, c in sorted(fam.items())], "family", [{"key": "count", "label": "Attempts", "color": "hold"}], "bar", note="Families missing here are the ones the model could not solve even once — training on this data will not fix them.")
R.chart("lengths", "Length of correct against wrong attempts", [{"bin": x["bin"], "correct": x["count"]} for x in hist(lengths_ok, bins=16)], "bin", [{"key": "correct", "label": "Correct attempts", "color": "kept"}], "bar")
R.table("kept", "A sample of what will be trained on", [{"key": "family", "label": "Family"}, {"key": "prompt", "label": "Question"}, {"key": "target", "label": "The model's own working"}], [{"family": r["family"], "prompt": r["prompt"][:180], "target": r["target"][:320]} for r in kept[:20]])
R.artifact(run_dir / "rft.jsonl", "rft.jsonl")
R.output("rft_train", str(run_dir / "rft.jsonl")).output("rft_val", str(run_dir / "rft_val.jsonl")).output("kept", len(kept)).output("coverage", solved)
for key in ("model", "tokenizer", "lang", "system"):
    R.output(key, I[key])
R.save()
