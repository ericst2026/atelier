"""Step 4 — contamination, then the card."""
import json
import os
import sys
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_nlp import hf
from atelier_nlp.dedup import shingles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_tasks import CHOICE  # noqa: E402

parse_args()
P = params({"shingle": 13, "corpus_docs": 40000, "notes": ""})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
task_dir = Path(I["task_dir"])
n = int(P["shingle"])

progress(5, f"indexing {int(P['corpus_docs']):,} training documents")
corpus = hf.dataset_split("pretrain", "wikitext-103", "train", limit=int(P["corpus_docs"]))
corpus_shingles = set()
for i, d in enumerate(corpus):
    corpus_shingles |= shingles(" ".join((d.get("text") or "").lower().split()), "word", n)
    if i % 5000 == 0:
        progress(5 + 45 * i / len(corpus), f"{i:,} documents · {len(corpus_shingles):,} phrases")

rows, per_benchmark = [], []
for bi, name in enumerate(I["benchmarks"]):
    items = read_jsonl(task_dir / f"{name}.jsonl")
    leaked = 0
    for x in items:
        text = x["question"] + " " + " ".join(x.get("choices", []))
        sh = shingles(" ".join(text.lower().split()), "word", n)
        overlap = sh & corpus_shingles
        if overlap:
            leaked += 1
            if len(rows) < 20:
                rows.append({"benchmark": name, "phrase": sorted(overlap)[0][:160], "question": x["question"][:180]})
    per_benchmark.append({"benchmark": name, "items": len(items), "leaked": leaked, "rate": leaked / max(len(items), 1)})
    progress(50 + 40 * (bi + 1) / len(I["benchmarks"]), f"{name}: {leaked}/{len(items)} items share a phrase with the corpus")

scores = json.loads(Path(I["scores"]).read_text()) if I.get("scores") else {"results": []}
sens = json.loads(Path(I["sensitivity"]).read_text()) if I.get("sensitivity") else []
worst_rate = max((b["rate"] for b in per_benchmark), default=0)
card = {
    "model": I.get("model"),
    "benchmarks": [{"benchmark": r["benchmark"], "method": r["method"], "accuracy": round(r["accuracy"], 4), "items": r["items"]} for r in scores.get("results", [])],
    "harness": {"shots": scores.get("shots"), "scoring": "log-probability per token for multiple choice, greedy generation for open answers", "sensitivity_spread": I.get("spread")},
    "contamination": per_benchmark,
    "notes": P["notes"],
    "caveats": [
        "Every number is for one harness. Changing the shot count or the scoring rule moves it.",
        "Contamination is measured against a stand-in corpus, not against what these models were actually trained on, which is not public for most of them.",
        "Accuracy on a few hundred items has a standard error of a few points; differences smaller than that are noise.",
    ],
}
(run_dir / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))

R = Result()
R.metric("contaminated", "Worst contamination rate", worst_rate, "pct", "dup" if worst_rate > 0.02 else "kept", help=f"items sharing a {n}-word phrase with the training corpus")
R.metric("items_checked", "Items checked", sum(b["items"] for b in per_benchmark), "int", "sky")
R.metric("phrases", "Phrases indexed", len(corpus_shingles), "int", "hold")
R.metric("spread", "Harness sensitivity", I.get("spread"), "pct", "raw", help="from step 3, carried into the card")
R.chart("contamination", "Contamination by benchmark", per_benchmark, "benchmark", [{"key": "rate", "label": "Items found in the corpus", "color": "dup"}], "bar", y_domain=[0, 1], note="This corpus is a stand-in. The models you scored were trained on far more of the web, so treat these rates as a floor, not an estimate.")
if scores.get("results"):
    R.chart("final", "The numbers you would publish", [{"benchmark": r["benchmark"], "accuracy": r["accuracy"]} for r in scores["results"] if r["method"] in ("mean", "generation")], "benchmark", [{"key": "accuracy", "label": "Accuracy", "color": "kept"}], "bar", y_domain=[0, 1])
if rows:
    R.table("leaks", "Benchmark items found in the corpus", [{"key": "benchmark", "label": "Benchmark"}, {"key": "phrase", "label": "Shared phrase"}, {"key": "question", "label": "The item"}], rows)
R.table("card", "The model card", [{"key": "field", "label": "Field"}, {"key": "value", "label": "Value"}], [
    {"field": "model", "value": str(card["model"])},
    {"field": "harness", "value": json.dumps(card["harness"], ensure_ascii=False)[:300]},
    {"field": "results", "value": "; ".join(f"{b['benchmark']} {b['accuracy']:.1%}" for b in card["benchmarks"][:8])},
    {"field": "contamination", "value": "; ".join(f"{b['benchmark']} {b['rate']:.1%}" for b in per_benchmark)},
    {"field": "notes", "value": card["notes"][:300]},
])
R.note("\n\n".join(["**Caveats recorded in the card**"] + [f"- {c}" for c in card["caveats"]]))
R.artifact(run_dir / "model_card.json", "model_card.json")
R.output("contamination", worst_rate).output("model_card", str(run_dir / "model_card.json"))
R.save()
