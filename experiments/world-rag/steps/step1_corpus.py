"""Step 1 — build the library, then find out how far keyword search gets."""
import json
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_mini.embed import BM25
from atelier_world import Library, World

parse_args()
P = params({"n_docs": 8000, "cluster_size": 4, "n_queries": 800, "n_pairs": 20000, "seed": 4242})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("base_run_run")
if not ref:
    raise SystemExit("Choose a Pretraining run (step 3) — the encoder is built from that model.")
o = ref["outputs"]
world = World(lang=o.get("lang", "en"), seed=int(P["seed"]))
lib = Library(world, n_docs=int(P["n_docs"]), seed=int(P["seed"]), cluster_size=int(P["cluster_size"]))
progress(25, f"{len(lib.documents):,} documents")
queries = lib.queries(int(P["n_queries"]))
pairs = lib.pairs(int(P["n_pairs"]))
progress(40, f"{len(queries)} evaluation queries, {len(pairs):,} training pairs")

texts = [d["text"] for d in lib.documents]
bm = BM25(texts)
ks = [1, 5, 20]
hits = {k: 0 for k in ks}
by_field = {}
hard = {}
for i, q in enumerate(queries):
    ranked = [idx for idx, _ in bm.search(q["query"], max(ks))]
    for k in ks:
        hits[k] += q["gold_id"] in ranked[:k]
    f = by_field.setdefault(q["field"], {k: 0 for k in ks} | {"n": 0})
    f["n"] += 1
    for k in ks:
        f[k] += q["gold_id"] in ranked[:k]
    if i % 100 == 0:
        progress(40 + 45 * i / len(queries), f"BM25 {i}/{len(queries)}")

# the documents keyword search wrongly returns become hard negatives for training
progress(88, "mining hard negatives")
for p in pairs:
    ranked = [idx for idx, _ in bm.search(p["query"], 3)]
    wrong = [idx for idx in ranked if idx != p["gold_id"]]
    if wrong:
        p["hard_negative"] = lib.documents[wrong[0]]["text"]
n_hard = sum(1 for p in pairs if p.get("hard_negative"))

write_jsonl(run_dir / "documents.jsonl", lib.documents)
write_jsonl(run_dir / "queries.jsonl", queries)
write_jsonl(run_dir / "pairs.jsonl", pairs)
(run_dir / "library.json").write_text(json.dumps({"n_docs": len(lib.documents), "cluster_size": int(P["cluster_size"]), "seed": int(P["seed"]), "lang": world.lang}, indent=2))
doc_lens = [len(t) for t in texts]

R = Result()
R.metric("documents", "Documents", len(lib.documents), "int", "kept")
R.metric("bm25_recall5", "BM25 recall@5", hits[5] / len(queries), "pct", "hold", help=f"recall@1 {hits[1] / len(queries):.1%}, recall@20 {hits[20] / len(queries):.1%}")
R.metric("hard_negatives", "Queries with a hard negative", n_hard / max(len(pairs), 1), "pct", "dup", help="a wrong document that keyword search ranked above or beside the right one")
R.metric("pairs", "Training pairs", len(pairs), "int", "sky")
R.chart("bm25", "Keyword search recall", [{"k": k, "recall": hits[k] / len(queries)} for k in ks], "k", [{"key": "recall", "label": "BM25", "color": "hold"}], "line", y_domain=[0, 1], note="This is the number to beat. A trained embedder that lands below this line has learned nothing worth the GPU time.")
R.chart("fields", "Recall@5 by what the query asks for", [{"field": f, "recall": v[5] / v["n"]} for f, v in sorted(by_field.items())], "field", [{"key": "recall", "label": "BM25 recall@5", "color": "hold"}], "bar", y_domain=[0, 1], note="Keyword search does best where the query happens to share a rare word with the document, and worst where the shared words point at the whole cluster.")
R.chart("lengths", "Document lengths", hist(doc_lens, bins=16), "bin", [{"key": "count", "label": "Documents", "color": "kept"}], "bar")
R.table("docs", "The library", [{"key": "name", "label": "Entity"}, {"key": "text", "label": "Document"}], [{"name": d["name"], "text": d["text"]} for d in lib.documents[:8]])
R.table("queries", "Queries and what keyword search returns", [{"key": "field", "label": "Asks for"}, {"key": "query", "label": "Query"}, {"key": "answer", "label": "Answer"}, {"key": "gold", "label": "Correct document"}, {"key": "bm25", "label": "BM25's first choice"}], [{"field": q["field"], "query": q["query"][:180], "answer": q["answer"], "gold": lib.documents[q["gold_id"]]["text"][:120], "bm25": lib.documents[bm.search(q["query"], 1)[0][0]]["text"][:120] if bm.search(q["query"], 1) else "—"} for q in queries[:12]], note="Read a few. The query and its document describe the same event in different words, and three other documents share most of the words that are left.")
R.artifact(run_dir / "documents.jsonl", "documents.jsonl").artifact(run_dir / "queries.jsonl", "queries.jsonl")
R.output("documents", str(run_dir / "documents.jsonl")).output("queries", str(run_dir / "queries.jsonl")).output("pairs", str(run_dir / "pairs.jsonl")).output("library", str(run_dir / "library.json"))
R.output("bm25_recall5", hits[5] / len(queries)).output("base_model", o["model"]).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en"))
R.save()
