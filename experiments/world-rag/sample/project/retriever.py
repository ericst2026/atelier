"""Your retriever."""
import numpy as np
import torch

from atelier_mini.embed import BM25, Index, MiniEmbedder
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import MiniTokenizer

_state = {}


def load_embedder(path: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MiniLM(MiniConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model_state"])
    tok = MiniTokenizer.load(ck["tokenizer"])
    emb = MiniEmbedder(model, tok, layer=ck["layer"], dim=ck["dim"] if ck["dim"] != model.config.n_embd else None, max_length=ck["max_length"]).to(device)
    if ck.get("proj") and isinstance(emb.proj, torch.nn.Linear):
        emb.proj.load_state_dict(ck["proj"])
    return emb


def build(documents: list[str]):
    """Called once with the whole corpus. Return whatever `search` needs."""
    emb = _state.get("embedder")
    lexical = BM25(documents)
    if emb is None:
        return {"lexical": lexical, "dense": None}
    vectors = emb.encode(documents, batch_size=256)
    return {"lexical": lexical, "dense": Index(vectors, [{"doc_id": i} for i in range(len(documents))]), "embedder": emb}


def search(index, queries: list[str], k: int = 5) -> list[list[int]]:
    if index.get("dense") is None:
        return [[i for i, _ in index["lexical"].search(q, k)] for q in queries]
    Q = index["embedder"].encode(queries, batch_size=256)
    dense = index["dense"].search(Q, k * 4)
    out = []
    for q, dres in zip(queries, dense):
        scores = {}
        for rank, r in enumerate(dres):
            scores[r["doc_id"]] = scores.get(r["doc_id"], 0.0) + 0.7 / (60 + rank)
        for rank, (i, _) in enumerate(index["lexical"].search(q, k * 4)):
            scores[i] = scores.get(i, 0.0) + 0.3 / (60 + rank)
        out.append([i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:k]])
    return out
