"""Step 1 — profile both sources on the same axes."""
import os
from pathlib import Path

from atelier_sdk import Result, hist, params, parse_args, progress
from atelier_nlp import hf

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_quality import signals  # noqa: E402

parse_args()
P = params({"n_docs": 40000, "sample_for_stats": 5000})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
n = int(P["n_docs"])
sources = {}
for key, group, name in (("fineweb", "curation", "fineweb-edu"), ("raw", "curation", "c4-raw")):
    rows = hf.dataset_split(group, name, "train", limit=n)
    sources[key] = rows
    progress(25 if key == "fineweb" else 50, f"{key}: {len(rows):,} documents")

stats, charts = {}, {}
for key, rows in sources.items():
    sample = rows[: int(P["sample_for_stats"])]
    sig = [signals(r["text"] or "") for r in sample]
    stats[key] = {k: sum(s[k] for s in sig) / len(sig) for k in sig[0]}
    charts[key] = sig
    progress(60 + 15 * (key == "raw"), f"profiled {key}")

axes = ["chars", "mean_word", "symbol_ratio", "digit_ratio", "dup_line_ratio", "unique_word_ratio", "ellipsis_ratio", "stopword_hits"]
compare = [{"signal": a, "fineweb": stats["fineweb"][a], "raw": stats["raw"][a]} for a in axes]
scores = [r.get("score") for r in sources["fineweb"] if isinstance(r.get("score"), (int, float))]

R = Result()
R.metric("docs", "Documents loaded", sum(len(v) for v in sources.values()), "int", "kept", help=f"{len(sources['fineweb']):,} filtered, {len(sources['raw']):,} raw")
R.metric("raw_chars", "Mean length, raw", stats["raw"]["chars"], "num", "raw", help=f"filtered: {stats['fineweb']['chars']:.0f}")
R.metric("raw_symbols", "Symbol ratio, raw", stats["raw"]["symbol_ratio"], "pct", "dup", help=f"filtered: {stats['fineweb']['symbol_ratio']:.1%}")
R.metric("raw_dup_lines", "Repeated lines, raw", stats["raw"]["dup_line_ratio"], "pct", "hold", help=f"filtered: {stats['fineweb']['dup_line_ratio']:.1%}")
R.chart("lengths", "Document lengths", [{"bin": a["bin"], "raw": a["count"], "fineweb": b["count"]} for a, b in zip(hist([s["chars"] for s in charts["raw"]], bins=20, log=True), hist([s["chars"] for s in charts["fineweb"]], bins=20, log=True))], "bin", [{"key": "raw", "label": "Raw", "color": "raw"}, {"key": "fineweb", "label": "Filtered", "color": "kept"}], "bar")
R.chart("symbols", "Symbol ratio", [{"bin": a["bin"], "raw": a["count"], "fineweb": b["count"]} for a, b in zip(hist([s["symbol_ratio"] for s in charts["raw"]], bins=20, lo=0, hi=0.6), hist([s["symbol_ratio"] for s in charts["fineweb"]], bins=20, lo=0, hi=0.6))], "bin", [{"key": "raw", "label": "Raw", "color": "raw"}, {"key": "fineweb", "label": "Filtered", "color": "kept"}], "bar", note="Where the two distributions overlap, no threshold will separate them — that is the part of curation a rule cannot do.")
if scores:
    R.chart("scores", "The classifier's quality score on the filtered source", hist(scores, bins=12), "bin", [{"key": "count", "label": "Documents", "color": "sky"}], "bar", note="This is the label your filter is implicitly trying to reproduce.")
R.chart("axes", "Mean of every signal, side by side", compare, "signal", [{"key": "raw", "label": "Raw", "color": "raw"}, {"key": "fineweb", "label": "Filtered", "color": "kept"}], "bar")
R.table("samples", "What raw text looks like", [{"key": "chars", "label": "Chars", "fmt": "int"}, {"key": "text", "label": "Text"}], [{"chars": len(r["text"] or ""), "text": (r["text"] or "")[:400]} for r in sources["raw"][:15]])
R.output("n_docs", n).output("fineweb_mean_chars", stats["fineweb"]["chars"])
R.note("Read the samples before touching a slider. Most of what a filter should remove is obvious on sight and hard to express as a threshold; the job is finding the thresholds that catch it without catching everything else.")
R.save()
