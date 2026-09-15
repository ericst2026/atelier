"""Step 2 — what the base model already does, under three prompt formats."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_world import World
from atelier_world.prepared import load_lm

parse_args()
P = params({"n_eval": 300, "max_new_tokens": 192, "formats": ["bare", "system", "fewshot"]})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.loads(Path(I["sft_meta"]).read_text())
world = World(lang=meta["lang"], seed=1, families=meta.get("families"))
lm = load_lm({"format": I.get("model_format", "atelier"), "model": I["base_model"], "tokenizer": I["tokenizer"], "adapter": I.get("adapter"), "system": I.get("system")}, device)
if meta.get("data_source") == "prepared":
    # the prepared dataset's held-out questions, and two of its training rows as examples
    tasks = read_jsonl(I["sft_val"], limit=int(P["n_eval"]))
    shots = read_jsonl(I["sft_train"], limit=2)
else:
    tasks = world.eval_set(int(P["n_eval"]), seed=555_003, families=meta.get("families"))
    shots = world.eval_set(2, seed=13)
fewshot = "\n\n".join(f"{s['prompt']}\n" + "\n".join(s["steps"]) + f"\n{world.answer_prefix} {s['answer']}" for s in shots)
formats = {
    "bare": (None, lambda q: q),
    "system": (world.system_prompt, lambda q: q),
    "fewshot": (world.system_prompt, lambda q: f"{fewshot}\n\n{q}"),
}
chosen = [f for f in P["formats"] if f in formats] or ["bare"]

results, rows, by_family = [], [], {}
for fi, name in enumerate(chosen):
    system, wrap = formats[name]
    prompts = [wrap(t["prompt"]) for t in tasks]
    outs = lm.generate(prompts, int(P["max_new_tokens"]), 0.0, batch_size=32, system=system, progress=lambda d, t: progress(5 + 90 * (fi + d / t) / len(chosen), f"{name}: {d}/{t}"))
    correct = 0
    marked = 0
    for t, o in zip(tasks, outs):
        ok = world.grade(o[0], t["answer"])
        correct += ok
        marked += world.answer_prefix in o[0]
        by_family.setdefault(t["family"], {}).setdefault(name, [0, 0])
        by_family[t["family"]][name][0] += ok
        by_family[t["family"]][name][1] += 1
        if name == chosen[-1] and len(rows) < 20:
            rows.append({"family": t["family"], "question": t["prompt"][:200], "gold": t["answer"], "model": o[0][:250], "ok": "✓" if ok else "✗"})
    results.append({"format": name, "accuracy": correct / len(tasks), "marked": marked / len(tasks)})

best = max(results, key=lambda r: r["accuracy"])
R = Result()
for r in results:
    R.metric(f"acc_{r['format']}", f"Accuracy · {r['format']}", r["accuracy"], "pct", "kept" if r is best else "sky", help=f"{r['marked']:.0%} wrote an answer line")
R.metric("best_baseline", "Best baseline", best["accuracy"], "pct", "hold", help=f"prompt format: {best['format']}")
R.chart("formats", "Accuracy by prompt format", [{"format": r["format"], "accuracy": r["accuracy"], "marked": r["marked"]} for r in results], "format", [{"key": "accuracy", "label": "Correct", "color": "kept"}, {"key": "marked", "label": "Wrote an answer line", "color": "sky"}], "bar", y_domain=[0, 1], note="A base model often knows the shape of an answer before it knows the answer. The gap between these two bars is what fine-tuning closes first.")
R.chart("families", "Accuracy by family", [dict({"family": f}, **{n: v[0] / v[1] for n, v in d.items()}) for f, d in sorted(by_family.items())], "family", [{"key": n, "label": n} for n in chosen], "bar", y_domain=[0, 1])
R.table("answers", "What it said", [{"key": "ok", "label": ""}, {"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "model", "label": "The model said"}], rows)
R.output("baseline_accuracy", best["accuracy"]).output("baseline_format", best["format"])
for k in ("sft_train", "sft_val", "sft_meta", "base_model", "tokenizer", "lang", "model_format", "adapter", "model_label"):
    if k in I:
        R.output(k, I[k])
R.save()
