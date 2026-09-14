"""Step 4 — accuracy before and after, and the reward-hacking checks."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_reward import repetition  # noqa: E402

parse_args()
P = params({"n_eval": 400, "max_new_tokens": 192, "held_out_families": True})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=5)
trained = list(I.get("families") or world.families)
families = world.families if bool(P["held_out_families"]) else trained
tasks = world.eval_set(int(P["n_eval"]), seed=99_881_003, families=families)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or world.system_prompt

out = {}
for j, (name, path) in enumerate((("before", I["policy"]), ("after", I["rl_model"]))):
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
        "answers": answers, "correct": correct,
        "accuracy": sum(correct) / len(tasks),
        "by_family": {k: v[0] / v[1] for k, v in by_family.items()},
        "marked": sum(1 for a in answers if world.answer_prefix in a) / len(tasks),
        "lengths": [len(a) for a in answers],
        "repetitive": sum(1 for a in answers if repetition(a) > 0.3) / len(tasks),
        "empty": sum(1 for a in answers if len(a.strip()) < 3) / len(tasks),
    }
    del model
    torch.cuda.empty_cache()

b, a = out["before"], out["after"]
trained_acc = ({"before": 0, "after": 0}, {"before": 0, "after": 0})
held = [f for f in a["by_family"] if f not in trained]
mean_len_b = sum(b["lengths"]) / len(tasks)
mean_len_a = sum(a["lengths"]) / len(tasks)
# marked-but-wrong is the sharpest hacking signal: the shape of an answer without the answer
wrong_marked_b = sum(1 for ans, c in zip(b["answers"], b["correct"]) if not c and world.answer_prefix in ans) / len(tasks)
wrong_marked_a = sum(1 for ans, c in zip(a["answers"], a["correct"]) if not c and world.answer_prefix in ans) / len(tasks)

R = Result()
R.metric("accuracy", "Accuracy after RL", a["accuracy"], "pct", "kept", help=f"{sum(a['correct'])} of {len(tasks)}")
R.metric("accuracy_before", "Accuracy before", b["accuracy"], "pct", "raw")
R.metric("gain", "Gained", a["accuracy"] - b["accuracy"], "pct", "sky")
R.metric("length_change", "Answer length change", (mean_len_a - mean_len_b) / max(mean_len_b, 1), "pct", "dup" if mean_len_a > mean_len_b * 1.3 else "hold", help=f"{mean_len_b:.0f} → {mean_len_a:.0f} characters")
R.metric("wrong_marked", "Confidently wrong", wrong_marked_a, "pct", "dup", help=f"marked an answer that was wrong; before: {wrong_marked_b:.0%}")
R.chart("overall", "Accuracy", [{"policy": "before", "accuracy": b["accuracy"]}, {"policy": "after", "accuracy": a["accuracy"]}], "policy", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("families", "Accuracy by family", [{"family": f, "before": b["by_family"].get(f, 0), "after": a["by_family"].get(f, 0), "trained": 1 if f in trained else 0} for f in sorted(a["by_family"])], "family", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1], note=("Families you did not train on: " + ", ".join(held) + ". If those fell while the trained ones rose, the policy narrowed rather than improved.") if held else None)
R.chart("hacking", "Signs of reward hacking", [
    {"signal": "wrong but marked", "before": wrong_marked_b, "after": wrong_marked_a},
    {"signal": "repetitive", "before": b["repetitive"], "after": a["repetitive"]},
    {"signal": "empty", "before": b["empty"], "after": a["empty"]},
], "signal", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "dup"}], "bar", note="All three should stay flat or fall. Any of them rising while reward rose means the reward paid for something other than being right.")
R.chart("lengths", "Answer length after RL", hist(a["lengths"], bins=18), "bin", [{"key": "count", "label": "Answers", "color": "hold"}], "bar")
changed = [i for i in range(len(tasks)) if a["correct"][i] != b["correct"][i]]
R.table("changed", "Questions whose outcome changed", [{"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "before", "label": "Before"}, {"key": "after", "label": "After"}], [{"family": tasks[i]["family"], "question": tasks[i]["prompt"][:180], "gold": tasks[i]["answer"], "before": ("✓ " if b["correct"][i] else "✗ ") + b["answers"][i][:180], "after": ("✓ " if a["correct"][i] else "✗ ") + a["answers"][i][:180]} for i in changed[:30]])
R.output("accuracy", a["accuracy"]).output("rl_model", I["rl_model"]).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en"))
R.save()
