"""Step 2 — apply the rules, count what each one costs."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_quality import default_keep, signals  # noqa: E402

parse_args()
P = params({"min_chars": 300, "max_chars": 100000, "min_mean_word": 3.0, "max_mean_word": 10.0, "max_symbol_ratio": 0.15, "max_digit_ratio": 0.2, "max_dup_line_ratio": 0.3, "min_stopword_hits": 2, "max_ellipsis_lines": 0.3})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
n = int(I.get("n_docs", 40000))
raw = hf.dataset_split("curation", "c4-raw", "train", limit=n)
reference = hf.dataset_split("curation", "fineweb-edu", "train", limit=min(n, 20000))
progress(10, f"{len(raw):,} raw documents")

kept, rejected = [], Counter()
for i, r in enumerate(raw):
    ok, why = default_keep(r["text"] or "", P)
    if ok:
        kept.append(r)
    else:
        rejected[why] += 1
    if i % 5000 == 0:
        progress(10 + 50 * i / len(raw), f"{i:,} filtered")

# the same rules applied to the already-filtered source: how much would you throw away?
ref_kept = sum(1 for r in reference if default_keep(r["text"] or "", P)[0])
false_positive = 1 - ref_kept / max(len(reference), 1)

# each rule on its own, so the cost of each is visible separately
solo = []
for rule in ("too short", "too long", "word length", "symbols", "digits", "repeated lines", "not prose", "truncated lines"):
    relaxed = dict(P)
    if rule == "too short":
        relaxed["min_chars"] = 0
    elif rule == "too long":
        relaxed["max_chars"] = 10**9
    elif rule == "word length":
        relaxed["min_mean_word"], relaxed["max_mean_word"] = 0, 100
    elif rule == "symbols":
        relaxed["max_symbol_ratio"] = 1
    elif rule == "digits":
        relaxed["max_digit_ratio"] = 1
    elif rule == "repeated lines":
        relaxed["max_dup_line_ratio"] = 1
    elif rule == "not prose":
        relaxed["min_stopword_hits"] = 0
    else:
        relaxed["max_ellipsis_lines"] = 1
    without = sum(1 for r in raw[:8000] if default_keep(r["text"] or "", relaxed)[0])
    with_all = sum(1 for r in raw[:8000] if default_keep(r["text"] or "", P)[0])
    solo.append({"rule": rule, "only_this_rule_removes": (without - with_all) / 8000, "removed_total": rejected[rule] / len(raw)})
progress(80, "measuring each rule")

hf.write_jsonl(run_dir / "filtered.jsonl", kept)
(run_dir / "filter_params.json").write_text(json.dumps(dict(P), indent=2))
kept_sig = [signals(r["text"] or "") for r in kept[:4000]]
dropped_sig = [signals(r["text"] or "") for r in raw[:4000] if not default_keep(r["text"] or "", P)[0]]

R = Result()
R.metric("kept", "Documents kept", len(kept), "int", "kept", help=f"{len(kept) / len(raw):.1%} of the raw source")
R.metric("removed", "Removed", len(raw) - len(kept), "int", "dup")
R.metric("false_positive", "Good documents your rules would remove", false_positive, "pct", "hold", help="the same rules applied to the already-filtered source")
R.metric("kept_chars", "Mean length of what you kept", sum(s["chars"] for s in kept_sig) / max(len(kept_sig), 1), "num", "sky", help=f"reference: {I.get('fineweb_mean_chars', 0):.0f}")
R.chart("rules", "What each rule removed", [{"rule": s["rule"], "removed": s["removed_total"], "uniquely": s["only_this_rule_removes"]} for s in solo], "rule", [{"key": "removed", "label": "Removed by this rule first", "color": "dup"}, {"key": "uniquely", "label": "Only this rule catches it", "color": "raw"}], "bar", note="A rule whose second bar is near zero is catching nothing the others miss — you can drop it and lose nothing but a threshold to tune.")
R.chart("funnel", "Survivors", [{"stage": "raw", "docs": len(raw)}, {"stage": "after filtering", "docs": len(kept)}], "stage", [{"key": "docs", "label": "Documents", "color": "kept"}], "bar")
R.chart("lengths", "Kept against dropped", [{"bin": a["bin"], "kept": a["count"], "dropped": b["count"]} for a, b in zip(hist([s["chars"] for s in kept_sig], bins=18, log=True), hist([s["chars"] for s in dropped_sig] or [1], bins=18, log=True))], "bin", [{"key": "kept", "label": "Kept", "color": "kept"}, {"key": "dropped", "label": "Dropped", "color": "dup"}], "bar")
R.table("dropped", "A sample of what you removed", [{"key": "why", "label": "Rule"}, {"key": "text", "label": "Text"}], [{"why": default_keep(r["text"] or "", P)[1], "text": (r["text"] or "")[:300]} for r in raw[:400] if not default_keep(r["text"] or "", P)[0]][:20], note="Read these. If good documents are in this table, a threshold is too tight.")
R.artifact(run_dir / "filtered.jsonl", "filtered.jsonl")
R.output("filtered", str(run_dir / "filtered.jsonl")).output("filter_params", str(run_dir / "filter_params.json")).output("kept_docs", len(kept)).output("n_docs", n)
R.save()
