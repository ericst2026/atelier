"""Step 4 — accuracy, drift, and the collapse check."""
import os
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_eval": 300, "max_new_tokens": 192})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=6)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or world.system_prompt
tasks = world.eval_set(int(P["n_eval"]), seed=505_117)

out = {}
for j, (name, path) in enumerate((("before", I["policy"]), ("after", I["dpo_model"]))):
    model, _ = MiniLM.load(path, device)
    gens = generate(model, tok, [t["prompt"] for t in tasks], int(P["max_new_tokens"]), 0.0, batch_size=32, system=system, progress=lambda d, t: progress(5 + 45 * j + 40 * d / t, f"{name}: {d}/{t}"))
    answers = [g[0] for g in gens]
    correct = [world.grade(a, t["answer"]) for a, t in zip(answers, tasks)]
    by_family = {}
    for c, t in zip(correct, tasks):
        f = by_family.setdefault(t["family"], [0, 0])
        f[0] += c
        f[1] += 1
    out[name] = {
        "answers": answers, "correct": correct, "accuracy": sum(correct) / len(tasks),
        "by_family": {k: v[0] / v[1] for k, v in by_family.items()},
        "lengths": [len(a) for a in answers],
        "empty": sum(1 for a in answers if len(a.strip()) < 3) / len(tasks),
        "marked": sum(1 for a in answers if world.answer_prefix in a) / len(tasks),
    }
    del model
    torch.cuda.empty_cache()

b, a = out["before"], out["after"]
families = sorted(set(b["by_family"]) | set(a["by_family"]))
mean_b, mean_a = sum(b["lengths"]) / len(tasks), sum(a["lengths"]) / len(tasks)

R = Result()
R.metric("accuracy", "Accuracy after DPO", a["accuracy"], "pct", "kept")
R.metric("accuracy_before", "Accuracy before", b["accuracy"], "pct", "raw")
R.metric("gain", "Gained", a["accuracy"] - b["accuracy"], "pct", "sky")
R.metric("empty", "Empty answers", a["empty"], "pct", "dup", help=f"before: {b['empty']:.1%} — the classic sign of policy collapse")
R.metric("length_change", "Answer length change", (mean_a - mean_b) / max(mean_b, 1), "pct", "hold", help=f"{mean_b:.0f} → {mean_a:.0f} characters")
R.chart("overall", "Accuracy", [{"model": "before", "accuracy": b["accuracy"]}, {"model": "after", "accuracy": a["accuracy"]}], "model", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("families", "Accuracy by family", [{"family": f, "before": b["by_family"].get(f, 0), "after": a["by_family"].get(f, 0)} for f in families], "family", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("health", "Signs of collapse", [{"signal": "empty answers", "before": b["empty"], "after": a["empty"]}, {"signal": "no answer line", "before": 1 - b["marked"], "after": 1 - a["marked"]}], "signal", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "dup"}], "bar", note="DPO can raise pair accuracy while making the model worse at producing anything. These two bars are where that shows up first.")
R.chart("lengths", "Answer lengths after DPO", hist(a["lengths"], bins=18), "bin", [{"key": "count", "label": "Answers", "color": "hold"}], "bar")
changed = [i for i in range(len(tasks)) if a["correct"][i] != b["correct"][i]]
R.table("changed", "What changed", [{"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "before", "label": "Before"}, {"key": "after", "label": "After"}], [{"family": tasks[i]["family"], "question": tasks[i]["prompt"][:170], "gold": tasks[i]["answer"], "before": ("✓ " if b["correct"][i] else "✗ ") + b["answers"][i][:180], "after": ("✓ " if a["correct"][i] else "✗ ") + a["answers"][i][:180]} for i in changed[:25]])
if I.get("sft_accuracy") is not None:
    R.note(f"The fine-tuned model this started from scored {float(I['sft_accuracy']):.1%}. Compare the reinforcement learning experiment on the same starting point: same goal, different mechanism.")
R.output("accuracy", a["accuracy"]).output("dpo_model", I["dpo_model"]).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en"))
R.save()
