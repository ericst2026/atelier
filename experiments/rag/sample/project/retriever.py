"""Your retriever."""
import numpy as np

from atelier_mini.embed import Embedder, Index, chunk_text
from atelier_nlp import hf

_state = {}


def build(corpus: list[dict]):
    """corpus: [{"title", "text"}]. Return whatever `search` needs."""
    emb = Embedder(hf.model_path("all-MiniLM-L6-v2"))
    chunks, payload = [], []
    for i, a in enumerate(corpus):
        for j, c in enumerate(chunk_text(a.get("text") or "", 400, 80)):
            chunks.append(c)
            payload.append({"doc_id": i, "title": a.get("title", ""), "text": c})
    vectors = emb.encode(chunks, batch_size=256)
    _state["emb"] = emb
    return Index(vectors, payload)


def search(index, question: str, k: int = 5) -> list[dict]:
    q = _state["emb"].encode([question])
    return index.search(q, k)[0]
