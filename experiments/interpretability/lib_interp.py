"""The four techniques, written plainly: hidden states, induction scores, the logit
lens, and a steering hook. Everything here works on any causal model from the
materials folder, because it finds the pieces by inspection rather than by name."""
from typing import Any, Callable, Optional

import numpy as np
import torch


# --- getting at the model's insides ----------------------------------------
def final_norm(model):
    """The normalisation applied just before the output embedding, whatever it is called."""
    for path in ("gpt_neox.final_layer_norm", "transformer.ln_f", "model.norm", "model.final_layernorm"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return torch.nn.Identity()


def blocks(model):
    """The list of transformer blocks."""
    for path in ("gpt_neox.layers", "transformer.h", "model.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("could not find the transformer blocks on this model")


@torch.no_grad()
def hidden_states(model, tok, texts: list[str], pooling: str = "last", batch_size: int = 16, max_length: int = 256, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """(examples, layers + 1, width). Index 0 is the embedding, the rest are blocks."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        res = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"]
        stacked = torch.stack(res.hidden_states, dim=1).float()          # (B, L+1, T, D)
        if pooling == "mean":
            m = mask[:, None, :, None].float()
            pooled = (stacked * m).sum(2) / m.sum(2).clamp(min=1)
        else:
            last = mask.sum(1) - 1
            pooled = stacked[torch.arange(stacked.shape[0], device=device), :, last, :]
        out.append(pooled.cpu().numpy())
        if progress:
            progress(min(i + batch_size, len(texts)), len(texts))
    return np.concatenate(out)


# --- probes -----------------------------------------------------------------
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
    Z = (X - probe["mu"]) / probe["sd"]
    return (Z @ probe["W"] + probe["b"]).argmax(1)


def probe_accuracy(probe: dict, X: np.ndarray, y: np.ndarray) -> float:
    return float((probe_predict(probe, X) == y).mean())


# --- induction heads --------------------------------------------------------
@torch.no_grad()
def induction_scores(model, tok, seq_len: int = 48, n_sequences: int = 16, seed: int = 0, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
    """(layers, heads). Feed a random sequence twice; score each head by how much
    attention lands on the token that followed the same token the first time round."""
    device = next(model.parameters()).device
    vocab = min(int(getattr(model.config, "vocab_size", 50000)), 20000)
    rng = np.random.default_rng(seed)
    scores = None
    for s in range(n_sequences):
        seq = rng.integers(100, vocab, size=seq_len)
        ids = torch.tensor([np.concatenate([seq, seq])], device=device)
        res = model(ids, output_attentions=True)
        # attention: tuple of (B, heads, T, T)
        att = torch.stack(res.attentions, dim=0)[:, 0].float()            # (L, H, T, T)
        # for query position seq_len + i, the induction target is position i + 1
        idx = torch.arange(seq_len - 1, device=device)
        q = seq_len + idx
        k = idx + 1
        picked = att[:, :, q, k]                                          # (L, H, seq_len-1)
        contribution = picked.mean(-1).cpu().numpy()
        scores = contribution if scores is None else scores + contribution
        if progress:
            progress(s + 1, n_sequences)
    return scores / n_sequences


@torch.no_grad()
def attention_pattern(model, tok, text: str, layer: int, head: int, max_length: int = 96):
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    res = model(**enc, output_attentions=True)
    att = res.attentions[layer][0, head].float().cpu().numpy()
    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    return att, [t.replace("Ġ", "␣").replace("Ċ", "⏎") for t in tokens]


# --- logit lens -------------------------------------------------------------
@torch.no_grad()
def logit_lens(model, tok, text: str, top_k: int = 5, apply_norm: bool = True, max_length: int = 128) -> list[dict]:
    """Decode the residual stream at every layer through the output embedding."""
    device = next(model.parameters()).device
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    res = model(**enc, output_hidden_states=True)
    head = model.get_output_embeddings()
    norm = final_norm(model) if apply_norm else torch.nn.Identity()
    final_top = int(res.logits[0, -1].argmax())
    rows = []
    for layer, h in enumerate(res.hidden_states):
        vec = h[0, -1]
        logits = head(norm(vec.unsqueeze(0))).float()[0]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, top_k)
        rows.append({
            "layer": layer,
            "tokens": [tok.decode([int(i)]) for i in top.indices],
            "probs": [float(p) for p in top.values],
            "final_rank": int((probs > probs[final_top]).sum()),
            "final_prob": float(probs[final_top]),
            "entropy": float(-(probs * (probs + 1e-12).log()).sum()),
        })
    return rows


# --- steering ---------------------------------------------------------------
@torch.no_grad()
def difference_vector(model, tok, positive: list[str], negative: list[str], layer: int, max_length: int = 64) -> np.ndarray:
    """Mean hidden state over one set minus the other, at one layer."""
    pos = hidden_states(model, tok, positive, "mean", max_length=max_length)[:, layer, :].mean(0)
    neg = hidden_states(model, tok, negative, "mean", max_length=max_length)[:, layer, :].mean(0)
    return pos - neg


class Steering:
    """Adds a vector into the residual stream at one block, for as long as it is open."""

    def __init__(self, model, layer: int, vector: np.ndarray, strength: float = 1.0):
        self.block = blocks(model)[layer]
        self.vector = torch.tensor(vector, dtype=torch.float32)
        self.strength = strength
        self.handle = None

    def __enter__(self):
        vec = self.vector.to(next(self.block.parameters()).device)

        def hook(module, args, output):
            if isinstance(output, tuple):
                hidden = output[0] + vec.to(output[0].dtype) * self.strength
                return (hidden,) + output[1:]
            return output + vec.to(output.dtype) * self.strength

        self.handle = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()
