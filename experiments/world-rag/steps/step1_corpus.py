"""Step 1 — build the library (generated, or prepared documents and questions), then find out how far keyword search gets."""
import json
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_mini.embed import BM25
from atelier_world import Library, World
from atelier_world.prepared import TEXT_FIELDS, _as_text, _data_files, _pick, _rows, choose_model, model_outputs, qa_row, read_documents, train_val

parse_args()
P = params({"model_source": "generated", "model_material": None, "data_source": "generated", "docs_material": None, "queries_material": None, "n_docs": 8000, "cluster_size": 4, "n_queries": 800, "n_pairs": 20000, "seed": 4242})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
base = choose_model(P, I, run_key="base_run", run_model_key="model", hint="Choose a Pretraining run (step 3), or a prepared model — the encoder is built from that model.")
lang = base["lang"]
prepared = P["data_source"] == "prepared"

# How a prepared question names the document that answers it (its gold document):
#   a key field  — doc_id / document_id / gold_id / doc / source / file / title — equal to a
#                  document's own id / doc_id / document_id / title / name / url field, its
#                  file name (with or without the extension), or, when the documents carry
#                  no id of their own, their position in the dataset (0 = the first);
#   a text field — context / passage / positive / document — holding the document's text.
# With no documents dataset chosen, the distinct context texts of the questions are the library.
QUERY_KEY_FIELDS = ("doc_id", "document_id", "gold_id", "doc", "source", "file", "title")
QUERY_TEXT_FIELDS = ("context", "passage", "positive", "document")
DOC_KEY_FIELDS = ("id", "doc_id", "document_id", "title", "name", "url")


def _norm(t) -> str:
    return " ".join(str(t).split()).lower()


def read_linked_qa(rel, split=None, limit=None):
    """read_qa, keeping the fields that name each row's gold document."""
    out = []
    for i, (_, row) in enumerate(_rows(_data_files(rel, split), limit)):
        q = qa_row(row, i)
        if q:
            q["raw"] = {k: row[k] for k in QUERY_KEY_FIELDS + QUERY_TEXT_FIELDS if row.get(k) not in (None, "", [], {})}
            out.append(q)
    if not out:
        raise SystemExit(f"materials/{rel} has no usable rows: each needs a question, an answer, and a field naming its document.")
    return out


def prepared_library():
    qrel, drel = P["queries_material"], P["docs_material"]
    if not qrel:
        raise SystemExit("Choose a prepared question dataset, or switch the library back to generated.")
    train_q, eval_q = train_val(qrel, read_linked_qa, int(P["n_queries"]), seed=int(P["seed"]), limit=int(P["n_pairs"]))
    docs: list[dict] = []
    by_key: dict = {}
    by_text: dict = {}
    if drel:
        raw_docs = read_documents(drel, limit=int(P["n_docs"]))
        # the JSONL rows once more, to recover each document's own id or title
        own: dict = {}
        for _, row in _rows(_data_files(drel, None), None):
            t = _as_text(_pick(row, TEXT_FIELDS))
            if t:
                own.setdefault(_norm(t), [str(row[k]) for k in DOC_KEY_FIELDS if row.get(k) not in (None, "", [], {})])
        files = Counter(d["file"] for d in raw_docs)
        has_ids = any(own.get(_norm(d["text"])) for d in raw_docs)
        keys: dict = {}
        for d in raw_docs:
            ks = list(own.get(_norm(d["text"]), []))
            if files[d["file"]] == 1:
                ks += [d["file"], Path(d["file"]).stem]
            if not has_ids:
                ks.append(str(d["id"]))
            for k in ks:
                keys.setdefault(_norm(k), set()).add(d["id"])
            name = next(iter(own.get(_norm(d["text"]), [])), None) or Path(d["file"]).stem
            docs.append({"id": d["id"], "name": name, "text": d["text"]})
            by_text.setdefault(_norm(d["text"]), d["id"])
        by_key = {k: next(iter(v)) for k, v in keys.items() if len(v) == 1}  # a key two documents share links nothing
    label = f"materials/{drel}" if drel else f"the context fields of materials/{qrel}"

    def link(q):
        for f in QUERY_KEY_FIELDS + QUERY_TEXT_FIELDS:
            v = q["raw"].get(f)
            if v is None:
                continue
            text = _norm(_as_text(v))
            if text in by_key:
                return by_key[text]
            if text in by_text:
                return by_text[text]
            if not drel and f in QUERY_TEXT_FIELDS and len(docs) < int(P["n_docs"]):
                docs.append({"id": len(docs), "name": f"context {len(docs)}", "text": _as_text(v)})
                by_text[text] = docs[-1]["id"]
                return docs[-1]["id"]
        return None

    def build(rows):
        out, lost = [], 0
        for q in rows:
            g = link(q)
            if g is None:
                lost += 1
                continue
            out.append({"id": len(out), "query": q["prompt"], "answer": q["answer"], "gold_id": g, "field": q["family"], "name": docs[g]["name"]})
        return out, lost

    queries, lost_q = build(eval_q)
    pairs, lost_p = build(train_q)
    if not queries or len(pairs) < 2:
        sample = eval_q[0]["raw"] if eval_q else {}
        raise SystemExit(
            f"Could not link the questions in materials/{qrel} to documents in {label}: {len(queries)} evaluation and {len(pairs)} training questions found their document. "
            f"Each question row needs a field naming its gold document — {' / '.join(QUERY_KEY_FIELDS)} equal to a document's id, title or file name, "
            f"or {' / '.join(QUERY_TEXT_FIELDS)} holding the document's text. "
            f"The first row's linking fields: {json.dumps(sample, ensure_ascii=False)[:300] if sample else 'none'}."
        )
    for p in pairs:
        p["positive"] = docs[p["gold_id"]]["text"]
    return docs, queries, pairs, lost_q + lost_p, label


if prepared:
    progress(10, f"linking materials/{P['queries_material']} to its documents")
    documents, queries, pairs, unlinked, data_label = prepared_library()
    progress(40, f"{len(documents):,} documents, {len(queries)} evaluation queries, {len(pairs):,} training pairs")
else:
    world = World(lang=lang, seed=int(P["seed"]))
    lib = Library(world, n_docs=int(P["n_docs"]), seed=int(P["seed"]), cluster_size=int(P["cluster_size"]))
    progress(25, f"{len(lib.documents):,} documents")
    documents = lib.documents
    queries = lib.queries(int(P["n_queries"]))
    pairs = lib.pairs(int(P["n_pairs"]))
    unlinked, data_label = 0, "generated from the world"
    progress(40, f"{len(queries)} evaluation queries, {len(pairs):,} training pairs")

texts = [d["text"] for d in documents]
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
        p["hard_negative"] = documents[wrong[0]]["text"]
n_hard = sum(1 for p in pairs if p.get("hard_negative"))

write_jsonl(run_dir / "documents.jsonl", documents)
write_jsonl(run_dir / "queries.jsonl", queries)
write_jsonl(run_dir / "pairs.jsonl", pairs)
(run_dir / "library.json").write_text(json.dumps({"n_docs": len(documents), "cluster_size": None if prepared else int(P["cluster_size"]), "seed": int(P["seed"]), "lang": lang, "data_source": P["data_source"], "data_label": data_label, "unlinked": unlinked}, indent=2))
doc_lens = [len(t) for t in texts]

R = Result()
R.metric("documents", "Documents", len(documents), "int", "kept", help=data_label)
R.metric("bm25_recall5", "BM25 recall@5", hits[5] / len(queries), "pct", "hold", help=f"recall@1 {hits[1] / len(queries):.1%}, recall@20 {hits[20] / len(queries):.1%}")
R.metric("hard_negatives", "Queries with a hard negative", n_hard / max(len(pairs), 1), "pct", "dup", help="a wrong document that keyword search ranked above or beside the right one")
R.metric("pairs", "Training pairs", len(pairs), "int", "sky")
R.chart("bm25", "Keyword search recall", [{"k": k, "recall": hits[k] / len(queries)} for k in ks], "k", [{"key": "recall", "label": "BM25", "color": "hold"}], "line", y_domain=[0, 1], note="This is the number to beat. A trained embedder that lands below this line has learned nothing worth the GPU time.")
R.chart("fields", "Recall@5 by what the query asks for", [{"field": f, "recall": v[5] / v["n"]} for f, v in sorted(by_field.items())], "field", [{"key": "recall", "label": "BM25 recall@5", "color": "hold"}], "bar", y_domain=[0, 1], note="Keyword search does best where the query happens to share a rare word with the document, and worst where the shared words point at the whole cluster.")
R.chart("lengths", "Document lengths", hist(doc_lens, bins=16), "bin", [{"key": "count", "label": "Documents", "color": "kept"}], "bar")
R.table("docs", "The library", [{"key": "name", "label": "Entity"}, {"key": "text", "label": "Document"}], [{"name": d["name"], "text": d["text"][:600]} for d in documents[:8]])
R.table("queries", "Queries and what keyword search returns", [{"key": "field", "label": "Asks for"}, {"key": "query", "label": "Query"}, {"key": "answer", "label": "Answer"}, {"key": "gold", "label": "Correct document"}, {"key": "bm25", "label": "BM25's first choice"}], [{"field": q["field"], "query": q["query"][:180], "answer": q["answer"], "gold": documents[q["gold_id"]]["text"][:120], "bm25": documents[bm.search(q["query"], 1)[0][0]]["text"][:120] if bm.search(q["query"], 1) else "—"} for q in queries[:12]], note="Read a few. The query and its document describe the same event in different words, and three other documents share most of the words that are left.")
R.artifact(run_dir / "documents.jsonl", "documents.jsonl").artifact(run_dir / "queries.jsonl", "queries.jsonl")
R.output("documents", str(run_dir / "documents.jsonl")).output("queries", str(run_dir / "queries.jsonl")).output("pairs", str(run_dir / "pairs.jsonl")).output("library", str(run_dir / "library.json"))
R.output("bm25_recall5", hits[5] / len(queries)).output("data_source", P["data_source"]).output("data_label", data_label)
for k, v in model_outputs(base, "base_model").items():
    R.output(k, v)
if prepared:
    R.note(
        f"The library is {data_label}. The evaluation queries are the validation or test split of materials/{P['queries_material']} (or a slice held out from it); the training pairs are the rest. "
        "Each question is linked to its gold document by a doc_id / document_id / gold_id / doc / source / file / title field matching a document's id, title or file name, or by a context / passage field holding its text."
        + (f" {unlinked} questions named no document that could be found and were left out." if unlinked else "")
        + f" Encoder base: {base['label']} ({base['format']})."
    )
else:
    R.note(f"Encoder base: {base['label']} ({base['format']}).")
R.save()
