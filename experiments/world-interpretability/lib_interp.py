"""The four techniques, on a MiniLM you trained yourself."""
from typing import Any, Callable, Optional

import numpy as np
import torch

from atelier_mini.tok import PAD


def _batch(tok, texts: list[str], device, max_length: int = 192):
    seqs = [tok.encode(t, bos=True)[:max_length] for t in texts]
    width = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), width), PAD, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), width), dtype=torch.float32, device=device)
    for i, s in enumerate(seqs):
        idx[i, : len(s)] = torch.tensor(s, device=device)
        mask[i, : len(s)] = 1.0
    return idx, mask


@torch.no_grad()
def hidden_states(model, tok, texts: list[str], pooling: str = "last", batch_size: int = 32, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """(examples, layers + 1, width). Index 0 is the embedding, the rest are blocks."""
    device = next(model.parameters()).device
    n_layers = len(model.blocks)
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        idx, mask = _batch(tok, chunk, device)
        per_layer = []
        for layer in range(n_layers + 1):
            h = model.hidden(idx, layer if layer < n_layers else -1).float()
            if pooling == "mean":
                pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            else:
                last = (mask.sum(1) - 1).long()
                pooled = h[torch.arange(h.shape[0], device=device), last]
            per_layer.append(pooled)
        out.append(torch.stack(per_layer, dim=1).cpu().numpy())
        if progress:
            progress(min(i + batch_size, len(texts)), len(texts))
    return np.concatenate(out)


def fit_probe(X: np.ndarray, y: np.ndarray, classes: int, steps: int = 300, lr: float = 0.05, weight_decay: float = 1e-3, seed: int = 0) -> dict:
    """Multinomial logistic regression by gradient descent — no scikit-learn needed."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    W = torch.zeros(Xt.shape[1], classes, device=device, requires_grad=True)
    b = torch.zeros(classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return {"W": W.detach().cpu().numpy(), "b": b.detach().cpu().numpy(), "mu": mu, "sd": sd, "loss": float(loss)}


def probe_predict(probe: dict, X: np.ndarray) -> np.ndarray:
    return ((X - probe["mu"]) / probe["sd"] @ probe["W"] + probe["b"]).argmax(1)


def probe_accuracy(probe: dict, X: np.ndarray, y: np.ndarray) -> float:
    return float((probe_predict(probe, X) == y).mean())


@torch.no_grad()
def induction_scores(model, tok, seq_len: int = 48, n_sequences: int = 16, seed: int = 0, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """(layers, heads). A random sequence repeated twice: score each head by how much
    attention the second copy pays to the token that followed the same token in the first."""
    device = next(model.parameters()).device
    vocab = model.config.vocab_size
    rng = np.random.default_rng(seed)
    total = None
    for s in range(n_sequences):
        seq = rng.integers(4, vocab, size=seq_len)
        ids = torch.tensor([np.concatenate([seq, seq])], device=device)
        att = model.attention_maps(ids)[:, 0].float()     # (L, H, T, T)
        idx = torch.arange(seq_len - 1, device=device)
        picked = att[:, :, seq_len + idx, idx + 1]
        contribution = picked.mean(-1).cpu().numpy()
        total = contribution if total is None else total + contribution
        if progress:
            progress(s + 1, n_sequences)
    return total / n_sequences


@torch.no_grad()
def logit_lens(model, tok, text: str, top_k: int = 5) -> list[dict]:
    """Decode every layer through the output embedding, which is the input embedding
    too — the weights are tied, so this is the same matrix the last layer uses."""
    device = next(model.parameters()).device
    ids = torch.tensor([tok.encode(text, bos=True)[: model.config.block_size]], device=device)
    final = model(ids)[0][0, -1].argmax().item()
    rows = []
    for layer in range(len(model.blocks) + 1):
        h = model.hidden(ids, layer if layer < len(model.blocks) else -1)[0, -1]
        logits = model.head(model.norm(h.unsqueeze(0))).float()[0]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, top_k)
        rows.append({
            "layer": layer,
            "tokens": [tok.decode([int(i)]) for i in top.indices],
            "probs": [float(p) for p in top.values],
            "final_rank": int((probs > probs[final]).sum()),
            "final_prob": float(probs[final]),
            "entropy": float(-(probs * (probs + 1e-12).log()).sum()),
        })
    return rows


@torch.no_grad()
def difference_vector(model, tok, positive: list[str], negative: list[str], layer: int) -> np.ndarray:
    pos = hidden_states(model, tok, positive, "mean")[:, layer, :].mean(0)
    neg = hidden_states(model, tok, negative, "mean")[:, layer, :].mean(0)
    return pos - neg


class Steering:
    """Adds a vector into the residual stream after one block, while it is open."""

    def __init__(self, model, layer: int, vector: np.ndarray, strength: float = 1.0):
        self.block = model.blocks[layer]
        self.vector = torch.tensor(vector, dtype=torch.float32)
        self.strength = strength
        self.handle = None

    def __enter__(self) -> "Steering":
        vec = self.vector.to(next(self.block.parameters()).device)

        def hook(module, args, output):
            return output + vec.to(output.dtype) * self.strength

        self.handle = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        if self.handle:
            self.handle.remove()
