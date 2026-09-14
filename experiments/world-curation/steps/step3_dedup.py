"""Step 3 — duplicates, which no per-document rule can find."""
import os
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl, write_jsonl
from atelier_nlp.dedup import dedupe

parse_args()
P = params({"near": True, "threshold": 0.75, "n": 5})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
docs = read_jsonl(I["filtered"])
texts = [d["text"] for d in docs]
progress(5, f"{len(docs):,} documents after filtering")

res = dedupe(texts, lowercase=True, collapse_ws=True, near=bool(P["near"]), shingle="word", n=int(P["n"]), threshold=float(P["threshold"]), progress=lambda pct, msg: progress(5 + 80 * pct / 100, msg))
kept_idx = set(res["kept"])
kept = [docs[i] for i in res["kept"]]
removed = [docs[i] for i in range(len(docs)) if i not in kept_idx]
write_jsonl(run_dir / "clean.jsonl", kept)

planted = [d for d in docs if d["kind"] == "duplicate"]
caught = sum(1 for d in removed if d["kind"] == "duplicate")
lost_good = sum(1 for d in removed if d["clean"])
st = res["stats"]

R = Result()
R.metric("dup_recall", "Planted duplicates caught", caught / max(len(planted), 1), "pct", "kept", help=f"{caught} of {len(planted)} that survived the filter")
R.metric("kept", "Documents kept", len(kept), "int", "raw")
R.metric("false_removals", "Clean documents removed", lost_good / max(len(docs), 1), "pct", "dup" if lost_good else "hold", help=f"{lost_good} documents that were not duplicates")
R.metric("chars_saved", "Characters removed", st["chars"] - st["chars_after_near"], "int", "sky")
R.chart("funnel", "What survived", [{"stage": "filtered", "docs": st["docs"]}, {"stage": "after exact", "docs": st["after_exact"]}, {"stage": "after near", "docs": st["after_near"]}], "stage", [{"key": "docs", "label": "Documents", "color": "kept"}], "bar")
if res["lsh"]:
    R.chart("sims", "Candidate pair similarity", res["sim_hist"], "bin", [{"key": "count", "label": "Pairs", "color": "sky"}], "bar", ref_x=f"{float(P['threshold']):.2f}", ref_label="threshold", note="Move the threshold and watch both numbers above move in opposite directions. There is no setting that catches every duplicate and removes nothing else.")
R.table("removed", "A sample of what went", [{"key": "kind", "label": "Kind"}, {"key": "text", "label": "Text"}], [{"kind": d["kind"], "text": d["text"][:220]} for d in removed[:20]])
R.artifact(run_dir / "clean.jsonl", "clean.jsonl")
R.output("clean", str(run_dir / "clean.jsonl")).output("corpus", I["corpus"]).output("dup_recall", caught / max(len(planted), 1))
for k in ("lang", "seed", "f1", "filter_params"):
    if k in I:
        R.output(k, I[k])
R.save()
