"""Sentence embeddings from a downloaded encoder, without sentence-transformers.

Mean-pool the last hidden states over the attention mask, then L2-normalise, which
is what the MiniLM and BGE checkpoints were trained to expect."""
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch


class Embedder:
    def __init__(self, path: str | Path, device: Optional[str] = None, max_length: int = 256, pooling: str = "mean"):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True).to(self.device).eval()
        self.max_length = max_length
        self.pooling = pooling
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64, prefix: str = "", progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [prefix + t for t in texts[i : i + batch_size]]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            if self.pooling == "cls":
                v = h[:, 0]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out.append(v.cpu().numpy())
            if progress:
                progress(min(i + batch_size, len(texts)), len(texts))
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)


class Index:
    """Brute-force cosine search. At a hundred thousand chunks this is milliseconds,
    and it has no index-building step to get wrong."""

    def __init__(self, vectors: np.ndarray, payload: list[dict]):
        self.vectors = vectors.astype(np.float32)
        self.payload = payload

    def search(self, queries: np.ndarray, k: int = 5) -> list[list[dict]]:
        sims = queries.astype(np.float32) @ self.vectors.T
        idx = np.argpartition(-sims, min(k, sims.shape[1] - 1), axis=1)[:, :k]
        out = []
        for row, cols in zip(sims, idx):
            ranked = sorted(cols, key=lambda c: -row[c])
            out.append([{**self.payload[c], "score": float(row[c]), "rank": r} for r, c in enumerate(ranked)])
        return out

    def save(self, path: Path):
        np.save(path, self.vectors)


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """Character chunks with an overlap, so a fact split across a boundary survives."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        cut = text.rfind(" ", start + size // 2, end)
        if cut == -1 or end == len(text):
            cut = end
        out.append(text[start:cut].strip())
        if cut >= len(text):
            break
        start = max(cut - overlap, start + 1)
    return [c for c in out if c]


def recall_at_k(results: list[list[dict]], gold_key: str, golds: list, k: int) -> float:
    hit = 0
    for res, gold in zip(results, golds):
        if any(r.get(gold_key) == gold for r in res[:k]):
            hit += 1
    return hit / max(len(golds), 1)


# --- embedding with a model you trained yourself -----------------------------
class MiniEmbedder(torch.nn.Module):
    """A MiniLM used as an encoder: mean-pool one layer's residual stream, project,
    normalise. With no projection and no training this is a mediocre but real
    embedder; trained contrastively it stops being mediocre."""

    def __init__(self, model, tok, layer: int = -1, dim: Optional[int] = None, max_length: int = 128):
        super().__init__()
        from .tok import PAD

        self.model = model
        self.tok = tok
        self.layer = layer
        self.max_length = max_length
        self.pad = PAD
        width = model.config.n_embd
        self.dim = dim or width
        self.proj = torch.nn.Linear(width, self.dim, bias=False) if dim and dim != width else torch.nn.Identity()
        if isinstance(self.proj, torch.nn.Linear):
            torch.nn.init.eye_(self.proj.weight) if self.dim == width else torch.nn.init.normal_(self.proj.weight, 0, width**-0.5)

    def _batch(self, texts: list[str], device):
        seqs = [self.tok.encode(t, bos=True)[: self.max_length] for t in texts]
        width = max(len(s) for s in seqs)
        idx = torch.full((len(seqs), width), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.float32)
        for i, s in enumerate(seqs):
            idx[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1.0
        return idx.to(device), mask.to(device)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        idx, mask = self._batch(texts, device)
        h = self.model.hidden(idx, self.layer)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return torch.nn.functional.normalize(self.proj(pooled).float(), dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 128, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        was = self.training
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(self(texts[i : i + batch_size]).cpu().numpy())
            if progress:
                progress(min(i + batch_size, len(texts)), len(texts))
        if was:
            self.train()
        return np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)


def train_contrastive(embedder, pairs: list[dict], out_dir, val_pairs: Optional[list[dict]] = None, epochs: float = 2.0, batch_size: int = 32, lr: float = 1e-4, temperature: float = 0.05, hard_negatives: bool = False, freeze_body: bool = False, device: Optional[str] = None, on_log: Optional[Callable[[dict], None]] = None) -> dict:
    """InfoNCE with in-batch negatives.

    Every other document in the batch is a negative for every query, so one batch of
    32 gives 31 negatives per example at no extra cost. That is the whole trick, and
    it is why the batch size matters more here than in ordinary training."""
    import random
    import time
    from pathlib import Path

    from .train import cosine_lr

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    if freeze_body:
        for p in embedder.model.parameters():
            p.requires_grad_(False)
    params = [p for p in embedder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = max(1, int(len(pairs) * epochs / batch_size))
    rng = random.Random(1)
    order = list(pairs)
    history, t0, cursor = [], time.time(), 0

    def evaluate(rows) -> float:
        """In-batch accuracy: does each query rank its own document first?"""
        embedder.eval()
        hits, n = 0, 0
        with torch.no_grad():
            for i in range(0, min(len(rows), 512), batch_size):
                chunk = rows[i : i + batch_size]
                if len(chunk) < 2:
                    continue
                q = embedder([c["query"] for c in chunk])
                d = embedder([c["positive"] for c in chunk])
                hits += int((q @ d.t()).argmax(1).eq(torch.arange(len(chunk), device=q.device)).sum())
                n += len(chunk)
        embedder.train()
        return hits / max(n, 1)

    embedder.train()
    for it in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, steps, lr, max(5, steps // 20))
        if it % max(10, steps // 20) == 0 or it == steps:
            row = {"step": it, "lr": opt.param_groups[0]["lr"], "elapsed": time.time() - t0}
            if val_pairs:
                row["in_batch_accuracy"] = evaluate(val_pairs)
            history.append(row)
            if on_log:
                on_log(row)
            if it == steps:
                break
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        chunk = order[cursor : cursor + batch_size]
        cursor += batch_size
        if len(chunk) < 2:
            continue
        q = embedder([c["query"] for c in chunk])
        docs = [c["positive"] for c in chunk]
        if hard_negatives:
            docs += [c["hard_negative"] for c in chunk if c.get("hard_negative")]
        d = embedder(docs)
        logits = q @ d.t() / temperature
        labels = torch.arange(len(chunk), device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if on_log and it % 5 == 0:
            on_log({"step": it, "loss": float(loss)})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": embedder.model.state_dict(), "config": embedder.model.config.to_dict(), "proj": embedder.proj.state_dict() if isinstance(embedder.proj, torch.nn.Linear) else None, "layer": embedder.layer, "dim": embedder.dim, "max_length": embedder.max_length}, out_dir / "embedder.pt")
    return {"history": history, "elapsed_sec": time.time() - t0, "checkpoint": str(out_dir / "embedder.pt"), "steps": steps}


class BM25:
    """Lexical search, for the baseline every embedder has to beat."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        import math
        from collections import Counter

        self.k1, self.b = k1, b
        self.docs = [self._tokenize(d) for d in documents]
        self.len = [len(d) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.docs), 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, d in enumerate(self.docs):
            for t in set(d):
                self.postings.setdefault(t, []).append(i)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        import re

        latin = re.findall(r"[\w']+", text.lower())
        cjk = [c for c in text if "\u3040" <= c <= "\u9fff"]
        return latin + cjk

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in self._tokenize(query):
            if t not in self.postings:
                continue
            idf = self.idf[t]
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / max(self.avg, 1e-9))
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / max(denom, 1e-9)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]
