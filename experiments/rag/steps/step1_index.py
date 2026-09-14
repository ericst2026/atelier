"""Step 1 — chunk the corpus, embed it, keep the vectors."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from atelier_sdk import Result, hist, params, parse_args, progress
from atelier_nlp import hf
from atelier_mini.embed import Embedder, chunk_text

parse_args()
P = params({"embedder": "all-MiniLM-L6-v2", "n_articles": 20000, "chunk_size": 400, "overlap": 80, "query_prefix": ""})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
articles = hf.dataset_split("rag", "wikipedia-simple", "train", limit=int(P["n_articles"]))
progress(5, f"{len(articles):,} articles")

chunks, payload = [], []
for i, a in enumerate(articles):
    text = a.get("text") or ""
    for j, c in enumerate(chunk_text(text, int(P["chunk_size"]), int(P["overlap"]))):
        chunks.append(c)
        payload.append({"doc_id": i, "title": a.get("title") or "", "chunk": j, "text": c})
    if i % 2000 == 0:
        progress(5 + 15 * i / len(articles), f"chunked {i:,} articles into {len(chunks):,} pieces")

emb = Embedder(hf.model_path(P["embedder"]))
t0 = time.time()
vectors = emb.encode(chunks, batch_size=256, progress=lambda d, t: progress(20 + 75 * d / t, f"embedded {d:,}/{t:,} chunks", step=d))
elapsed = time.time() - t0
np.save(run_dir / "vectors.npy", vectors)
hf.write_jsonl(run_dir / "chunks.jsonl", payload)
(run_dir / "index_meta.json").write_text(json.dumps({"embedder": P["embedder"], "dim": emb.dim, "chunk_size": int(P["chunk_size"]), "overlap": int(P["overlap"]), "chunks": len(chunks), "articles": len(articles), "query_prefix": P["query_prefix"]}, indent=2))

lengths = [len(c) for c in chunks]
per_article = {}
for p in payload:
    per_article[p["doc_id"]] = per_article.get(p["doc_id"], 0) + 1
R = Result()
R.metric("chunks", "Chunks indexed", len(chunks), "int", "kept", help=f"from {len(articles):,} articles")
R.metric("dim", "Embedding dimension", emb.dim, "int", "sky", help=P["embedder"])
R.metric("index_mb", "Index size", vectors.nbytes / 1e6, "num", "hold", help="float32 vectors, before any compression")
R.metric("embed_rate", "Chunks embedded per second", len(chunks) / max(elapsed, 1e-9), "num", "raw")
R.chart("lengths", "Chunk lengths", hist(lengths, bins=20), "bin", [{"key": "count", "label": "Chunks", "color": "kept"}], "bar")
R.chart("per_article", "Chunks per article", hist(list(per_article.values()), bins=16, integer=True), "bin", [{"key": "count", "label": "Articles", "color": "sky"}], "bar")
R.table("samples", "What a chunk looks like", [{"key": "title", "label": "Article"}, {"key": "text", "label": "Chunk"}], [{"title": p["title"], "text": p["text"][:300]} for p in payload[:15]])
R.artifact(run_dir / "vectors.npy", "vectors.npy").artifact(run_dir / "chunks.jsonl", "chunks.jsonl")
R.output("vectors", str(run_dir / "vectors.npy")).output("chunks", str(run_dir / "chunks.jsonl")).output("index_meta", str(run_dir / "index_meta.json")).output("embedder", P["embedder"]).output("n_chunks", len(chunks))
R.note("Everything downstream is capped by this step. A fact that landed in no chunk, or in a chunk with nothing around it to make it findable, cannot be retrieved later by any model.")
R.save()
