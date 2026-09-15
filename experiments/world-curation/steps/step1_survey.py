"""Step 1 — a corpus whose damage is labelled (generated), or a prepared one without labels, and a look at it."""
import os
import random
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, params, parse_args, progress, write_jsonl
from atelier_world import World, make_corpus
from atelier_world.prepared import read_documents

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_signals import signals  # noqa: E402

parse_args()
P = params({"data_source": "generated", "data_material": None, "docs": 20000, "dirty_share": 0.55, "duplicate_share": 0.12, "lang": "en", "seed": 17, "heldout_share": 0.1})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
world = World(lang=P["lang"], seed=int(P["seed"]))
prepared = P["data_source"] == "prepared"
axes = ["chars", "symbol_ratio", "digit_ratio", "upper_ratio", "dup_line_ratio", "unique_word_ratio", "stopword_hits", "ellipsis_ratio"]

if prepared:
    if not P.get("data_material"):
        raise SystemExit("Choose a prepared corpus from the list, or switch the corpus back to generated.")
    data_label = f"materials/{P['data_material']}"
    progress(10, f"reading {data_label}")
    raw = read_documents(str(P["data_material"]), limit=int(P["docs"]))
    # nobody labelled this text: `clean` is unknown, and `kind` is only the file it came from
    random.Random(int(P["seed"])).shuffle(raw)
    n_held = min(3000, max(1, int(len(raw) * float(P["heldout_share"]))))
    if len(raw) < 20:
        raise SystemExit(f"{data_label} has {len(raw)} documents; curation needs at least 20 (some are held out to validate the model in step 4).")
    held, raw = raw[:n_held], raw[n_held:]
    docs = [{"id": i, "text": d["text"], "kind": d["kind"], "clean": None} for i, d in enumerate(raw)]
    # held-out documents step 4 validates on; no corpus it trains sees them
    write_jsonl(run_dir / "heldout.jsonl", [{"id": i, "text": d["text"], "kind": d["kind"]} for i, d in enumerate(held)])
else:
    data_label = "generated from the world"
    progress(10, f"generating {int(P['docs']):,} documents and spoiling some of them")
    docs = make_corpus(world, int(P["docs"]), seed=int(P["seed"]), dirty_share=float(P["dirty_share"]), duplicate_share=float(P["duplicate_share"]))
write_jsonl(run_dir / "corpus.jsonl", docs)
progress(50, "profiling")

sample = docs[:6000]
sig = {d["id"]: signals(d["text"]) for d in sample}
kinds = Counter(d["kind"] for d in docs)


def means(pred) -> dict:
    rows = [sig[d["id"]] for d in sample if pred(d)]
    return {a: sum(r[a] for r in rows) / max(len(rows), 1) for a in axes} if rows else {a: 0 for a in axes}


per_kind = [{"kind": k, **{a: round(means(lambda d, k=k: d["kind"] == k)[a], 4) for a in ("symbol_ratio", "digit_ratio", "upper_ratio", "unique_word_ratio")}} for k in sorted(kinds)]
progress(85, "writing result")

R = Result()
if prepared:
    q = lambda a, f: sorted(sig[d["id"]][a] for d in sample)[min(len(sample) - 1, int(f * len(sample)))]  # noqa: E731
    R.metric("docs", "Documents", len(docs), "int", "raw", help=f"{data_label}, plus {len(held)} held out for step 4")
    R.metric("labels", "Labels", "none", "text", "dup", help="a prepared corpus says nothing about which documents are spoiled, so precision and recall cannot be measured")
    R.metric("chars", "Characters", sum(len(d["text"]) for d in docs), "int", "sky")
    R.chart("axes", "Every signal across your corpus", [{"signal": a, "p10": q(a, 0.1), "median": q(a, 0.5), "p90": q(a, 0.9)} for a in axes if a != "chars"], "signal", [{"key": "p10", "label": "10th percentile", "color": "kept"}, {"key": "median", "label": "Median", "color": "raw"}, {"key": "p90", "label": "90th percentile", "color": "dup"}], "bar", note="With no labels there is no clean-against-spoiled comparison. Look for the tails: a 90th percentile far from the median is where a threshold would bite.")
    R.chart("lengths", "Document lengths", hist([sig[d["id"]]["chars"] for d in sample], bins=18, log=True), "bin", [{"key": "count", "label": "Documents", "color": "raw"}], "bar")
    R.table("kinds", "Signals by source file", [{"key": "kind", "label": "File"}, {"key": "symbol_ratio", "label": "Symbols", "fmt": "num"}, {"key": "digit_ratio", "label": "Digits", "fmt": "num"}, {"key": "upper_ratio", "label": "Capitals", "fmt": "num"}, {"key": "unique_word_ratio", "label": "Distinct words", "fmt": "num"}], per_kind[:50])
    R.table("samples", "The first documents", [{"key": "kind", "label": "File"}, {"key": "text", "label": "Text"}], [{"kind": d["kind"], "text": d["text"][:260]} for d in docs[:12]])
    R.artifact(run_dir / "corpus.jsonl", "corpus.jsonl").artifact(run_dir / "heldout.jsonl", "heldout.jsonl")
    R.output("corpus", str(run_dir / "corpus.jsonl")).output("heldout", str(run_dir / "heldout.jsonl")).output("lang", P["lang"]).output("seed", int(P["seed"])).output("docs", len(docs))
    R.output("data_source", "prepared").output("data_label", data_label)
    R.note(f"Your own corpus, {data_label}. It has no labels, so the later steps can show what each rule and the dedup pass remove but not whether they were right: precision, recall, F1, duplicates caught and the perfectly-clean ceiling are all unavailable. The training step still answers the question that matters — does the filtered corpus train a better model — on {len(held)} documents held out here.")
    R.save()
    raise SystemExit(0)

clean_share = sum(d["clean"] for d in docs) / len(docs)
clean_m, dirty_m = means(lambda d: d["clean"]), means(lambda d: not d["clean"])
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
R.output("data_source", "generated").output("data_label", data_label)
R.note("Read the samples before touching a slider. Duplicates are the interesting case: every copy is good text, so nothing you can measure about one document will find them. That is the next step but one.")
R.save()
