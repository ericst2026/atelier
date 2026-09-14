"""Step 4 — exact-match accuracy per family, base against fine-tuned."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_eval": 300, "max_new_tokens": 192, "temperature": 0.0})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.loads(Path(I["sft_meta"]).read_text())
world = World(lang=meta["lang"], seed=1, families=meta.get("families"))
tok = MiniTokenizer.load(I["tokenizer"])
tasks = world.eval_set(int(P["n_eval"]), seed=555_003, families=meta.get("families"))
system = I.get("system") or meta["system"]

out = {}
for j, (name, path) in enumerate((("base", I["base_model"]), ("tuned", I["sft_model"]))):
    model, _ = MiniLM.load(path, device)
    gens = generate(model, tok, [t["prompt"] for t in tasks], int(P["max_new_tokens"]), float(P["temperature"]), batch_size=32, system=system, progress=lambda d, t: progress(5 + 45 * j + 40 * d / t, f"{name}: {d}/{t}"))
    answers = [g[0] for g in gens]
    correct = [world.grade(a, t["answer"]) for a, t in zip(answers, tasks)]
    by_family = {}
    for c, t in zip(correct, tasks):
        f = by_family.setdefault(t["family"], [0, 0])
        f[0] += c
        f[1] += 1
    by_diff = {}
    for c, t in zip(correct, tasks):
        d = by_diff.setdefault(t["difficulty"], [0, 0])
        d[0] += c
        d[1] += 1
    out[name] = {
        "answers": answers, "correct": correct,
        "accuracy": sum(correct) / len(tasks),
        "by_family": {k: v[0] / v[1] for k, v in by_family.items()},
        "by_diff": {k: v[0] / v[1] for k, v in by_diff.items()},
        "marked": sum(1 for a in answers if world.answer_prefix in a) / len(tasks),
        "lengths": [len(a) for a in answers],
    }
    del model
    torch.cuda.empty_cache()

b, t_ = out["base"], out["tuned"]
families = sorted(set(b["by_family"]) | set(t_["by_family"]))
diffs = sorted(set(b["by_diff"]) | set(t_["by_diff"]))
fixed = [i for i in range(len(tasks)) if t_["correct"][i] and not b["correct"][i]]
broken = [i for i in range(len(tasks)) if b["correct"][i] and not t_["correct"][i]]

R = Result()
R.metric("accuracy", "Accuracy after fine-tuning", t_["accuracy"], "pct", "kept", help=f"{sum(t_['correct'])} of {len(tasks)}")
R.metric("accuracy_before", "Accuracy before", b["accuracy"], "pct", "raw")
R.metric("gain", "Gained", t_["accuracy"] - b["accuracy"], "pct", "sky", help=f"{len(fixed)} newly correct, {len(broken)} lost")
R.metric("marked", "Wrote an answer line", t_["marked"], "pct", "hold", help=f"before: {b['marked']:.0%}")
R.metric("answer_chars", "Answer length (characters)", sum(t_["lengths"]) / len(tasks), "num", "sky", help=f"before: {sum(b['lengths']) / len(tasks):.0f}")
R.chart("overall", "Accuracy", [{"model": "base", "accuracy": b["accuracy"]}, {"model": "fine-tuned", "accuracy": t_["accuracy"]}], "model", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("families", "Accuracy by task family", [{"family": f, "before": b["by_family"].get(f, 0), "after": t_["by_family"].get(f, 0)} for f in families], "family", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1], note="A family still at zero needs more demonstrations of that family, not more steps.")
R.chart("difficulty", "Accuracy by difficulty", [{"difficulty": str(d), "before": b["by_diff"].get(d, 0), "after": t_["by_diff"].get(d, 0)} for d in diffs], "difficulty", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("lengths", "Answer lengths after fine-tuning", hist(t_["lengths"], bins=18), "bin", [{"key": "count", "label": "Answers", "color": "hold"}], "bar")
R.table("fixed", "Questions fine-tuning fixed", [{"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "base", "label": "Base said"}, {"key": "tuned", "label": "Fine-tuned said"}], [{"family": tasks[i]["family"], "question": tasks[i]["prompt"][:200], "gold": tasks[i]["answer"], "base": b["answers"][i][:200], "tuned": t_["answers"][i][:200]} for i in fixed[:20]])
if broken:
    R.table("broken", "Questions it lost", [{"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "tuned", "label": "Fine-tuned said"}], [{"family": tasks[i]["family"], "question": tasks[i]["prompt"][:200], "gold": tasks[i]["answer"], "tuned": t_["answers"][i][:220]} for i in broken[:15]], note="Worth reading: this is usually a family that is under-represented in the demonstrations.")
R.output("accuracy", t_["accuracy"]).output("sft_model", I["sft_model"]).output("tokenizer", I["tokenizer"]).output("lang", meta["lang"]).output("system", system)
R.save()
