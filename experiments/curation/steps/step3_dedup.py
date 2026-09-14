"""Step 3 — near-duplicate removal, then the contamination check."""
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf
from atelier_nlp.dedup import dedupe, shingles

parse_args()
P = params({"near": True, "threshold": 0.7, "n": 8, "contamination_n": 13})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
docs = hf.read_jsonl(I["filtered"])
texts = [d["text"] or "" for d in docs]
progress(5, f"{len(docs):,} documents")

res = dedupe(texts, lowercase=True, collapse_ws=True, near=bool(P["near"]), shingle="word", n=int(P["n"]), threshold=float(P["threshold"]), progress=lambda pct, msg: progress(5 + 55 * pct / 100, msg))
kept_idx = set(res["kept"])
kept = [docs[i] for i in res["kept"]]
st = res["stats"]

# contamination: does the training data contain the text we will evaluate on?
progress(70, "checking for contamination against the validation set")
val = hf.dataset_split("pretrain", "wikitext-103", "validation", limit=3000)
val_shingles = set()
for v in val:
    val_shingles |= shingles(" ".join((v["text"] or "").lower().split()), "word", int(P["contamination_n"]))
leaked, leak_rows = 0, []
for i, d in enumerate(kept):
    sh = shingles(" ".join((d["text"] or "").lower().split()), "word", int(P["contamination_n"]))
    overlap = sh & val_shingles
    if overlap:
        leaked += 1
        if len(leak_rows) < 15:
            leak_rows.append({"overlap": len(overlap), "phrase": sorted(overlap)[0][:180], "text": (d["text"] or "")[:220]})
    if i % 2000 == 0:
        progress(70 + 20 * i / max(len(kept), 1), f"contamination {i:,}/{len(kept):,}")
clean = [d for i, d in enumerate(kept) if d not in ()]  # keep everything; contamination is reported, not silently removed
hf.write_jsonl(run_dir / "clean.jsonl", kept)

R = Result()
R.metric("kept", "Documents kept", st["after_near"], "int", "kept", help=f"from {st['docs']:,}")
R.metric("removed", "Duplicates removed", st["docs"] - st["after_near"], "int", "dup", help=f"{len(res['removed_exact'])} exact, {len(res['removed_near'])} near")
R.metric("chars_saved", "Characters saved", st["chars"] - st["chars_after_near"], "int", "hold", help=f"{1 - st['chars_after_near'] / max(st['chars'], 1):.1%} of the corpus")
R.metric("contaminated", "Documents overlapping the validation set", leaked / max(len(kept), 1), "pct", "dup" if leaked else "kept", help=f"{leaked} documents share a {P['contamination_n']}-word phrase with the held-out text")
R.chart("funnel", "What survived", [{"stage": "filtered", "docs": st["docs"], "chars": st["chars"]}, {"stage": "after exact", "docs": st["after_exact"], "chars": st["chars_after_exact"]}, {"stage": "after near", "docs": st["after_near"], "chars": st["chars_after_near"]}], "stage", [{"key": "docs", "label": "Documents", "color": "kept"}, {"key": "chars", "label": "Characters", "color": "sky", "axis": "right"}], "bar")
if res["lsh"]:
    R.chart("sims", "Candidate pair similarity", res["sim_hist"], "bin", [{"key": "count", "label": "Pairs", "color": "sky"}], "bar", ref_x=f"{float(P['threshold']):.2f}", ref_label="threshold")
    R.chart("clusters", "Cluster sizes", res["cluster_sizes"], "bin", [{"key": "count", "label": "Clusters", "color": "dup"}], "bar", note="A large cluster is one page republished many times. Leaving them in teaches the model that page.")
if leak_rows:
    R.table("leaks", "Contamination found", [{"key": "overlap", "label": "Shared phrases", "fmt": "int"}, {"key": "phrase", "label": "A shared phrase"}, {"key": "text", "label": "The training document"}], leak_rows, note="Any of this in your corpus makes the validation loss look better than the model is. Decide deliberately whether to remove these documents or change the validation set.")
R.artifact(run_dir / "clean.jsonl", "clean.jsonl")
R.output("clean", str(run_dir / "clean.jsonl")).output("contamination", leaked / max(len(kept), 1)).output("clean_docs", len(kept))
for k in ("filtered", "filter_params", "n_docs"):
    if k in I:
        R.output(k, I[k])
R.save()
