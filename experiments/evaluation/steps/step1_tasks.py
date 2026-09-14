"""Step 1 — load the benchmarks and look at what is actually in them."""
import os
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, params, parse_args, progress, write_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_tasks import CHOICE, load  # noqa: E402

parse_args()
P = params({"benchmarks": ["hellaswag", "arc-easy", "arc-challenge", "mmlu", "gsm8k"], "limit": 500, "seed": 1})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
names = [b for b in P["benchmarks"]] or ["arc-easy"]
loaded, rows, summary = {}, [], []
for i, name in enumerate(names):
    try:
        items = load(name, int(P["limit"]), int(P["seed"]))
    except Exception as exc:
        summary.append({"benchmark": name, "items": 0, "kind": "unavailable", "note": str(exc)[:120]})
        progress(10 + 80 * (i + 1) / len(names), f"{name}: not installed")
        continue
    loaded[name] = items
    kinds = Counter(x["kind"] for x in items)
    lengths = [len(x["question"]) for x in items]
    choice_items = [x for x in items if x["kind"] == CHOICE]
    # the cheapest possible cheat: is the right answer simply the longest option?
    longest_correct = sum(1 for x in choice_items if x["choices"] and max(range(len(x["choices"])), key=lambda j: len(x["choices"][j])) == x["answer"]) / max(len(choice_items), 1)
    summary.append({"benchmark": name, "items": len(items), "kind": list(kinds)[0], "mean_question_chars": sum(lengths) / len(lengths), "choices": len(choice_items[0]["choices"]) if choice_items else 0, "longest_is_correct": longest_correct, "subjects": len({x.get("subject") for x in items})})
    for x in items[:4]:
        rows.append({"benchmark": name, "question": x["question"][:220], "options": " | ".join(x.get("choices", [])[:4])[:220] if x["kind"] == CHOICE else "(generated)", "answer": x["choices"][x["answer"]][:80] if x["kind"] == CHOICE else x["answer"]})
    progress(10 + 80 * (i + 1) / len(names), f"{name}: {len(items)} items")
    write_jsonl(run_dir / f"{name}.jsonl", items)

total = sum(s["items"] for s in summary)
R = Result()
R.metric("items", "Items loaded", total, "int", "kept", help=f"{len([s for s in summary if s['items']])} benchmarks")
R.metric("benchmarks", "Benchmarks available", len([s for s in summary if s["items"]]), "int", "sky")
worst = max((s for s in summary if s.get("longest_is_correct") is not None), key=lambda s: s.get("longest_is_correct", 0), default=None)
if worst:
    R.metric("longest_is_correct", "Longest option is correct", worst.get("longest_is_correct", 0), "pct", "dup" if worst.get("longest_is_correct", 0) > 0.35 else "hold", help=f"{worst['benchmark']} — chance would be about {1 / max(worst.get('choices', 4), 1):.0%}")
R.chart("sizes", "Items per benchmark", [{"benchmark": s["benchmark"], "items": s["items"]} for s in summary], "benchmark", [{"key": "items", "label": "Items", "color": "kept"}], "bar")
R.chart("bias", "How often the longest option is the right one", [{"benchmark": s["benchmark"], "longest": s.get("longest_is_correct", 0), "chance": 1 / max(s.get("choices", 4) or 4, 1)} for s in summary if s["items"]], "benchmark", [{"key": "longest", "label": "Longest option correct", "color": "dup"}, {"key": "chance", "label": "Chance", "color": "hold"}], "bar", y_domain=[0, 1], note="Where the first bar clears the second, a model that prefers long answers scores above chance without understanding anything. This is why length normalisation is a real decision.")
R.table("summary", "The benchmarks", [{"key": "benchmark", "label": "Benchmark"}, {"key": "items", "label": "Items", "fmt": "int"}, {"key": "kind", "label": "Shape"}, {"key": "choices", "label": "Options", "fmt": "int"}, {"key": "subjects", "label": "Subjects", "fmt": "int"}, {"key": "mean_question_chars", "label": "Question chars", "fmt": "num"}, {"key": "note", "label": "Note"}], summary)
R.table("items", "What the items look like", [{"key": "benchmark", "label": "Benchmark"}, {"key": "question", "label": "Question"}, {"key": "options", "label": "Options"}, {"key": "answer", "label": "Correct"}], rows)
R.output("benchmarks", [s["benchmark"] for s in summary if s["items"]]).output("task_dir", str(run_dir)).output("limit", int(P["limit"]))
R.note("Read a dozen items before you score anything. Every benchmark contains items whose answer is arguable, and knowing roughly how many changes how you read a two-point difference between models.")
R.save()
