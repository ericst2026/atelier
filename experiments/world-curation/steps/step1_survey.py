"""Step 1 — generate a corpus whose damage is labelled, and look at it."""
import os
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, params, parse_args, progress, write_jsonl
from atelier_world import World, make_corpus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_signals import signals  # noqa: E402

parse_args()
P = params({"docs": 20000, "dirty_share": 0.55, "duplicate_share": 0.12, "lang": "en", "seed": 17})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
world = World(lang=P["lang"], seed=int(P["seed"]))
progress(10, f"generating {int(P['docs']):,} documents and spoiling some of them")
docs = make_corpus(world, int(P["docs"]), seed=int(P["seed"]), dirty_share=float(P["dirty_share"]), duplicate_share=float(P["duplicate_share"]))
write_jsonl(run_dir / "corpus.jsonl", docs)
progress(50, "profiling")

sample = docs[:6000]
sig = {d["id"]: signals(d["text"]) for d in sample}
kinds = Counter(d["kind"] for d in docs)
clean_share = sum(d["clean"] for d in docs) / len(docs)
axes = ["chars", "symbol_ratio", "digit_ratio", "upper_ratio", "dup_line_ratio", "unique_word_ratio", "stopword_hits", "ellipsis_ratio"]


def means(pred) -> dict:
    rows = [sig[d["id"]] for d in sample if pred(d)]
    return {a: sum(r[a] for r in rows) / max(len(rows), 1) for a in axes} if rows else {a: 0 for a in axes}


clean_m, dirty_m = means(lambda d: d["clean"]), means(lambda d: not d["clean"])
per_kind = [{"kind": k, **{a: round(means(lambda d, k=k: d["kind"] == k)[a], 4) for a in ("symbol_ratio", "digit_ratio", "upper_ratio", "unique_word_ratio")}} for k in sorted(kinds)]
progress(85, "writing result")

R = Result()
R.metric("docs", "Documents", len(docs), "int", "raw")
R.metric("clean_share", "Clean", clean_share, "pct", "kept", help=f"{len(docs) - sum(d['clean'] for d in docs)} spoiled or duplicated")
R.metric("kinds", "Kinds of damage", len(kinds) - 1, "int", "dup")
R.metric("chars", "Characters", sum(len(d["text"]) for d in docs), "int", "sky")
R.chart("kinds", "Documents by kind", [{"kind": k, "count": v} for k, v in kinds.most_common()], "kind", [{"key": "count", "label": "Documents", "color": "dup"}], "bar", note="Each kind is a different problem. Some are obvious on one axis, some are only visible by comparing documents with each other.")
R.chart("axes", "Clean against spoiled, on every signal", [{"signal": a, "clean": clean_m[a], "spoiled": dirty_m[a]} for a in axes], "signal", [{"key": "clean", "label": "Clean", "color": "kept"}, {"key": "spoiled", "label": "Spoiled", "color": "dup"}], "bar", note="Where the two bars differ, a threshold can separate them. Where they do not, no rule on a single document will.")
R.chart("lengths", "Document lengths", [{"bin": a["bin"], "clean": a["count"], "spoiled": b["count"]} for a, b in zip(hist([sig[d["id"]]["chars"] for d in sample if d["clean"]], bins=18, log=True), hist([sig[d["id"]]["chars"] for d in sample if not d["clean"]] or [1], bins=18, log=True))], "bin", [{"key": "clean", "label": "Clean", "color": "kept"}, {"key": "spoiled", "label": "Spoiled", "color": "dup"}], "bar")
R.table("kinds", "How each kind looks on four axes", [{"key": "kind", "label": "Kind"}, {"key": "symbol_ratio", "label": "Symbols", "fmt": "num"}, {"key": "digit_ratio", "label": "Digits", "fmt": "num"}, {"key": "upper_ratio", "label": "Capitals", "fmt": "num"}, {"key": "unique_word_ratio", "label": "Distinct words", "fmt": "num"}], per_kind)
R.table("samples", "One of each", [{"key": "kind", "label": "Kind"}, {"key": "keep", "label": "Should keep"}, {"key": "text", "label": "Text"}], [{"kind": k, "keep": "yes" if next(d for d in docs if d["kind"] == k)["clean"] else "no", "text": next(d for d in docs if d["kind"] == k)["text"][:260]} for k in sorted(kinds)])
R.artifact(run_dir / "corpus.jsonl", "corpus.jsonl")
R.output("corpus", str(run_dir / "corpus.jsonl")).output("lang", P["lang"]).output("seed", int(P["seed"])).output("docs", len(docs))
R.note("Read the samples before touching a slider. Duplicates are the interesting case: every copy is good text, so nothing you can measure about one document will find them. That is the next step but one.")
R.save()
