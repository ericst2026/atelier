"""Step 2 — exact + near-duplicate removal with MinHash/LSH."""
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl, write_jsonl
from atelier_nlp.dedup import dedupe

parse_args()
P = params({"lowercase": True, "collapse_ws": True, "strip_punct": False, "near": True, "shingle": "word", "n": 3, "threshold": 0.7, "keep": "first"})
I = inputs()
corpus = Path(I["corpus"])
docs = read_jsonl(corpus)
texts = [d["text"] for d in docs]
progress(5, f"{len(docs)} documents loaded")

res = dedupe(texts, bool(P["lowercase"]), bool(P["collapse_ws"]), bool(P["strip_punct"]), bool(P["near"]), P["shingle"], int(P["n"]), float(P["threshold"]), P["keep"], progress=progress)
kept_idx = set(res["kept"])
removed = []
for i, j in res["removed_exact"].items():
    removed.append({**docs[i], "removed_by": "exact", "dup_of": docs[j]["id"], "similarity": 1.0, "kept_text": docs[j]["text"]})
for i, (j, s) in res["removed_near"].items():
    removed.append({**docs[i], "removed_by": "near", "dup_of": docs[j]["id"], "similarity": round(s, 3), "kept_text": docs[j]["text"]})
removed.sort(key=lambda r: (-r["similarity"], r["id"]))

run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
kept_path, removed_path = run_dir / "kept.jsonl", run_dir / "removed.jsonl"
write_jsonl(kept_path, [docs[i] for i in res["kept"]])
write_jsonl(removed_path, removed)

planted = [d for d in docs if d.get("truth") in ("exact", "near")]
kept_ids = {docs[i]["id"] for i in kept_idx}
# a planted pair counts as caught when either member was removed (with "keep first", the copy may be the survivor)
planted_caught = sum(1 for d in planted if d["id"] not in kept_ids or d.get("dup_of") not in kept_ids)
originals = {d.get("dup_of") for d in planted}
unplanned = sum(1 for r in removed if r.get("truth") == "unique" and r["id"] not in originals)
st = res["stats"]

R = Result()
R.metric("kept", "Kept", st["after_near"], "int", "kept")
R.metric("removed", "Removed", st["docs"] - st["after_near"], "int", "dup", help=f"{len(res['removed_exact'])} exact, {len(res['removed_near'])} near")
if planted:
    R.metric("planted_recall", "Planted duplicates caught", planted_caught / len(planted), "pct", "kept", help=f"{planted_caught} of {len(planted)}")
    R.metric("unplanned", "Unplanned removals", unplanned, "int", "hold", help="Documents removed that were not planted — either real duplicates in the source or false positives")
R.metric("candidates", "Candidate pairs", res["candidates"], "int", "sky", help=f"LSH bands×rows = {res['lsh']['bands']}×{res['lsh']['rows']}" if res["lsh"] else None)
funnel = [
    {"stage": "input", "docs": st["docs"], "chars": st["chars"]},
    {"stage": "after exact", "docs": st["after_exact"], "chars": st["chars_after_exact"]},
    {"stage": "after near", "docs": st["after_near"], "chars": st["chars_after_near"]},
]
R.chart("funnel", "What survived each pass", funnel, "stage", [{"key": "docs", "label": "Documents", "color": "kept"}, {"key": "chars", "label": "Characters", "color": "sky", "axis": "right"}], "bar", note="Documents on the left axis, characters on the right.")
if res["lsh"]:
    R.chart("sims", "Candidate pair similarity", res["sim_hist"], "bin", [{"key": "count", "label": "Pairs", "color": "sky"}], "bar", ref_x=f"{float(P['threshold']):.2f}", ref_label="threshold", note="Pairs at or right of the line were merged into one cluster.")
    R.chart("clusters", "Cluster sizes", res["cluster_sizes"], "bin", [{"key": "count", "label": "Clusters", "color": "dup"}], "bar")
R.chart("by_source", "Removed by source", [{"source": s, "count": c} for s, c in Counter(r["source"] for r in removed).most_common()], "source", [{"key": "count", "label": "Removed", "color": "dup"}], "bar")
R.table("removed", "What was removed", [{"key": "id", "label": "#"}, {"key": "removed_by", "label": "Pass"}, {"key": "similarity", "label": "Jaccard", "fmt": "num"}, {"key": "truth", "label": "Planted"}, {"key": "text", "label": "Removed text"}, {"key": "kept_text", "label": "Kept text"}], [{"id": r["id"], "removed_by": r["removed_by"], "similarity": r["similarity"], "truth": "" if r["truth"] == "unique" else r["truth"], "text": r["text"][:200], "kept_text": r["kept_text"][:200]} for r in removed[:60]], note="Side by side with the document that was kept.")
R.artifact(kept_path, "kept.jsonl").artifact(removed_path, "removed.jsonl")
R.output("corpus", str(kept_path)).output("removed", str(removed_path)).output("docs", st["after_near"])
R.save()
