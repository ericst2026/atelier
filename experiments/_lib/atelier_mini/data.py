"""Packing text into one flat token stream, and sampling batches from it."""
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


def pack(texts: Iterable[str], tok, out_path: Path, eos: bool = True, max_tokens: int | None = None, progress=None) -> dict:
    """Concatenate documents with <eos> between them into a uint16/uint32 array."""
    dtype = np.uint16 if tok.vocab_size < 65_535 else np.uint32
    buf: list[np.ndarray] = []
    n = 0
    docs = 0
    lengths = []
    # a report every so many documents, or every second, whichever comes first: a
    # small corpus would otherwise finish before its first report
    last = time.monotonic()
    for text in texts:
        ids = tok.encode(text, eos=eos)
        lengths.append(len(ids))
        if max_tokens and n + len(ids) > max_tokens:
            ids = ids[: max_tokens - n]
        buf.append(np.array(ids, dtype=dtype))
        n += len(ids)
        docs += 1
        if progress and (docs % 5000 == 0 or time.monotonic() - last > 1.0):
            progress(docs, n)
            last = time.monotonic()
        if max_tokens and n >= max_tokens:
            break
    if progress and docs:
        progress(docs, n)
    arr = np.concatenate(buf) if buf else np.zeros(0, dtype=dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_path)
    return {"tokens": int(arr.size), "docs": docs, "dtype": dtype.__name__, "path": str(out_path), "doc_lengths": lengths}


class TokenStream:
    def __init__(self, path: str | Path, dtype: str = "uint16"):
        self.data = np.memmap(path, dtype=np.uint16 if dtype == "uint16" else np.uint32, mode="r")

    def __len__(self) -> int:
        return int(self.data.size)

    def batch(self, batch_size: int, block_size: int, device, generator: torch.Generator | None = None):
        ix = torch.randint(len(self.data) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def sft_batch(rows: list[dict], tok, block_size: int, device, system: str | None = None):
    """Build (idx, targets, mask) where the mask is 1 only on the target tokens.
    The model reads the question but is never scored on predicting it."""
    from .tok import EOS, PAD

    seqs, masks = [], []
    for r in rows:
        prompt = (f"{system}\n{r['prompt']}\n" if system else f"{r['prompt']}\n")
        p_ids = tok.encode(prompt, bos=True)
        t_ids = tok.encode(r["target"], eos=True)
        ids = (p_ids + t_ids)[:block_size]
        mask = ([0] * len(p_ids) + [1] * len(t_ids))[:block_size]
        seqs.append(ids)
        masks.append(mask)
    width = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), width), PAD, dtype=torch.long)
    msk = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        idx[i, : len(s)] = torch.tensor(s)
        msk[i, : len(m)] = torch.tensor(m)
    x, y = idx[:, :-1], idx[:, 1:].clone()
    mask = msk[:, 1:]
    y[mask == 0] = -100
    return x.to(device), y.to(device), mask.to(device)
