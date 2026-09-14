"""Step 2 — score them, and choose how."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_score import option_logprobs, pick  # noqa: E402
from lib_tasks import CHOICE, GENERATION, few_shot_prefix  # noqa: E402

parse_args()
P = params({"model": "SmolLM2-360M", "scoring": ["sum", "mean", "unconditional"], "shots": 0, "max_new_tokens": 256, "batch_size": 8})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
task_dir = Path(I["task_dir"])
benchmarks = list(I["benchmarks"])
mp = hf.model_path(P["model"])
tok = hf.load_tokenizer(mp, padding_side="right")
model = hf.load_model(mp)
methods = [m for m in P["scoring"] if m in ("sum", "mean", "unconditional")] or ["mean"]

results, per_item = [], []
for bi, name in enumerate(benchmarks):
    items = read_jsonl(task_dir / f"{name}.jsonl")
    if not items:
        continue
    prefix = few_shot_prefix(items, int(P["shots"]))
    if items[0]["kind"] == CHOICE:
        picks = {m: [] for m in methods}
        for i, x in enumerate(items):
            scores = option_logprobs(model, tok, prefix + x["question"], x["choices"], int(P["batch_size"]))
            for m in methods:
                picks[m].append(pick(scores, m))
            if i % 25 == 0:
                progress(5 + 85 * (bi + i / len(items)) / len(benchmarks), f"{name}: {i}/{len(items)}")
            if bi == 0 and i < 12:
                per_item.append({"benchmark": name, "question": x["question"][:150], "correct": x["choices"][x["answer"]][:70], "chose": x["choices"][picks[methods[0]][-1]][:70], "ok": "✓" if picks[methods[0]][-1] == x["answer"] else "✗"})
        for m in methods:
            acc = sum(1 for p, x in zip(picks[m], items) if p == x["answer"]) / len(items)
            results.append({"benchmark": name, "method": m, "accuracy": acc, "chance": 1 / len(items[0]["choices"]), "items": len(items)})
    else:
        prompts = [hf.chat_prompt(tok, prefix + x["question"] + "\nSolve step by step, then give the final answer after '####'.") for x in items]
        gens = hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), 0.0, batch_size=max(4, int(P["batch_size"])), progress=lambda d, t: progress(5 + 85 * (bi + d / t) / len(benchmarks), f"{name}: {d}/{t}"))
        acc = sum(hf.answers_equal(hf.extract_answer(g[0]), x["answer"]) for g, x in zip(gens, items)) / len(items)
        results.append({"benchmark": name, "method": "generation", "accuracy": acc, "chance": 0.0, "items": len(items)})

(run_dir / "scores.json").write_text(json.dumps({"model": P["model"], "shots": int(P["shots"]), "results": results}, indent=2))
merged = {}
for r in results:
    merged.setdefault(r["benchmark"], {"benchmark": r["benchmark"], "chance": r["chance"]})[r["method"]] = r["accuracy"]
main = [r for r in results if r["method"] in (methods[0], "generation")]
mean_acc = sum(r["accuracy"] for r in main) / max(len(main), 1)
spread = max((max(v.get(m, 0) for m in methods if m in v) - min(v.get(m, 1) for m in methods if m in v)) for v in merged.values() if any(m in v for m in methods)) if len(methods) > 1 else 0.0

R = Result()
R.metric("accuracy", "Mean accuracy", mean_acc, "pct", "kept", help=f"{P['model']}, {P['shots']}-shot")
R.metric("method_spread", "Spread across scoring methods", spread, "pct", "dup" if spread > 0.05 else "hold", help="the same model and items, scored three defensible ways")
R.metric("above_chance", "Benchmarks above chance", sum(1 for r in main if r["accuracy"] > r["chance"] + 0.05), "int", "sky", help=f"of {len(main)}")
R.chart("accuracy", "Accuracy by benchmark and scoring method", list(merged.values()), "benchmark", [{"key": m, "label": m} for m in methods] + [{"key": "chance", "label": "Chance", "color": "hold"}], "bar", y_domain=[0, 1], note="Total log-probability favours short options; the per-token version favours long ones. Published numbers rarely say which was used.")
R.chart("gap", "Accuracy above chance", [{"benchmark": r["benchmark"], "above": r["accuracy"] - r["chance"]} for r in main], "benchmark", [{"key": "above", "label": "Above chance", "color": "kept"}], "bar")
R.table("results", "Every number", [{"key": "benchmark", "label": "Benchmark"}, {"key": "method", "label": "Scoring"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "chance", "label": "Chance", "fmt": "pct"}, {"key": "items", "label": "Items", "fmt": "int"}], results)
R.table("items", "A sample of the decisions", [{"key": "ok", "label": ""}, {"key": "benchmark", "label": "Benchmark"}, {"key": "question", "label": "Question"}, {"key": "correct", "label": "Correct"}, {"key": "chose", "label": "The model chose"}], per_item)
R.artifact(run_dir / "scores.json", "scores.json")
R.output("scores", str(run_dir / "scores.json")).output("model", P["model"]).output("accuracy", mean_acc)
for k in ("task_dir", "benchmarks", "limit"):
    R.output(k, I[k])
R.save()
