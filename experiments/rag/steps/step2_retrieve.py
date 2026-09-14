"""Step 2 — recall@k, with and without a reranker."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf
from atelier_mini.embed import Embedder, Index

parse_args()
P = params({"n_questions": 400, "k_values": ["1", "3", "5", "10", "20"], "rerank": True, "rerank_candidates": 30, "include_unanswerable": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
meta = json.loads(Path(I["index_meta"]).read_text())
chunks = hf.read_jsonl(I["chunks"])
vectors = np.load(I["vectors"])
index = Index(vectors, chunks)

# SQuAD gives the paragraph that answers each question; we look for that text in the index
squad = hf.dataset_split("rag", "squad-v2", "validation", limit=6000)
answerable = [q for q in squad if (q.get("answers") or {}).get("text")]
unanswerable = [q for q in squad if not (q.get("answers") or {}).get("text")]
n = int(P["n_questions"])
questions = answerable[:n]
progress(5, f"{len(questions)} answerable questions, {len(unanswerable)} without an answer")

# a retrieved chunk counts as correct when it overlaps the gold paragraph substantially
def gold_hit(chunk_text_: str, context: str) -> bool:
    a = " ".join(chunk_text_.lower().split())
    b = " ".join((context or "").lower().split())
    if not a or not b:
        return False
    probe = a[: min(len(a), 120)]
    return probe in b or a in b


emb = Embedder(hf.model_path(meta["embedder"]))
t0 = time.time()
q_vecs = emb.encode([q["question"] for q in questions], batch_size=128, prefix=meta.get("query_prefix", ""), progress=lambda d, t: progress(10 + 25 * d / t, f"embedded {d}/{t} questions"))
embed_ms = (time.time() - t0) / max(len(questions), 1) * 1000
ks = sorted(int(k) for k in P["k_values"])
max_k = max(ks + [int(P["rerank_candidates"]) if bool(P["rerank"]) else 0])
t0 = time.time()
results = index.search(q_vecs, max_k)
search_ms = (time.time() - t0) / max(len(questions), 1) * 1000
progress(45, "scoring recall")

hits = [[gold_hit(r["text"], q.get("context", "")) for r in res] for q, res in zip(questions, results)]
rows = [{"k": k, "recall": sum(1 for h in hits if any(h[:k])) / len(hits), "method": "embedding"} for k in ks]

rerank_rows, rerank_ms = [], 0.0
if bool(P["rerank"]):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rpath = hf.model_path("ms-marco-MiniLM-L-6-v2")
    rtok = AutoTokenizer.from_pretrained(str(rpath), local_files_only=True)
    rmodel = AutoModelForSequenceClassification.from_pretrained(str(rpath), local_files_only=True).to(emb.device).eval()
    cand = int(P["rerank_candidates"])
    t0 = time.time()
    reranked_hits = []
    for qi, (q, res) in enumerate(zip(questions, results)):
        pairs = [(q["question"], r["text"]) for r in res[:cand]]
        scores = []
        for i in range(0, len(pairs), 32):
            batch = pairs[i : i + 32]
            enc = rtok([p[0] for p in batch], [p[1] for p in batch], padding=True, truncation=True, max_length=384, return_tensors="pt").to(emb.device)
            with torch.no_grad():
                scores += rmodel(**enc).logits.squeeze(-1).float().tolist()
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        reranked_hits.append([gold_hit(res[i]["text"], q.get("context", "")) for i in order])
        if qi % 25 == 0:
            progress(50 + 40 * qi / len(questions), f"reranked {qi}/{len(questions)}")
    rerank_ms = (time.time() - t0) / max(len(questions), 1) * 1000
    rerank_rows = [{"k": k, "recall": sum(1 for h in reranked_hits if any(h[:k])) / len(reranked_hits), "method": "reranked"} for k in ks if k <= cand]

merged = {}
for r in rows:
    merged.setdefault(r["k"], {"k": r["k"]})["embedding"] = r["recall"]
for r in rerank_rows:
    merged.setdefault(r["k"], {"k": r["k"]})["reranked"] = r["recall"]
curve = [merged[k] for k in sorted(merged)]
r5 = next((r["recall"] for r in rows if r["k"] == 5), rows[-1]["recall"])
best5 = next((r["recall"] for r in rerank_rows if r["k"] == 5), r5)
hf.write_jsonl(run_dir / "questions.jsonl", [{**q, "retrieved": [{"text": r["text"], "doc_id": r["doc_id"], "title": r["title"], "score": r["score"]} for r in res[:10]]} for q, res in zip(questions, results)])
if bool(P["include_unanswerable"]):
    hf.write_jsonl(run_dir / "unanswerable.jsonl", unanswerable[:500])

R = Result()
R.metric("recall_5", "Recall@5", r5, "pct", "kept", help=f"with reranking: {best5:.1%}" if rerank_rows else None)
R.metric("recall_1", "Recall@1", rows[0]["recall"], "pct", "sky")
R.metric("recall_ceiling", f"Recall@{ks[-1]}", rows[-1]["recall"], "pct", "hold", help="the ceiling for anything downstream — a reranker cannot find what search missed")
R.metric("latency", "Milliseconds per question", embed_ms + search_ms + rerank_ms, "num", "raw", help=f"{embed_ms:.1f} embedding + {search_ms:.1f} search" + (f" + {rerank_ms:.1f} reranking" if rerank_ms else ""))
R.chart("recall", "Recall against k", curve, "k", [{"key": "embedding", "label": "Embedding only", "color": "sky"}, {"key": "reranked", "label": "With a reranker", "color": "kept"}], "line", y_domain=[0, 1], note="Reranking reorders what search already found; the two lines must meet at the candidate limit.")
R.chart("latency", "Where the time goes", [{"stage": "embed the question", "ms": embed_ms}, {"stage": "search", "ms": search_ms}] + ([{"stage": "rerank", "ms": rerank_ms}] if rerank_ms else []), "stage", [{"key": "ms", "label": "Milliseconds", "color": "raw"}], "bar")
R.table("misses", "Questions where the right passage was not in the top 5", [{"key": "question", "label": "Question"}, {"key": "gold", "label": "Should have found"}, {"key": "found", "label": "Found instead"}], [{"question": q["question"][:160], "gold": (q.get("context") or "")[:200], "found": res[0]["text"][:200]} for q, res, h in zip(questions, results, hits) if not any(h[:5])][:15], note="Read these before blaming the model. Most are questions whose wording shares no vocabulary with the passage that answers them.")
R.artifact(run_dir / "questions.jsonl", "questions.jsonl")
R.output("questions", str(run_dir / "questions.jsonl")).output("unanswerable", str(run_dir / "unanswerable.jsonl")).output("recall_5", r5)
for k in ("vectors", "chunks", "index_meta", "embedder"):
    R.output(k, I[k])
R.save()
