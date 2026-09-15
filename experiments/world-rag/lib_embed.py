"""Embedders for this experiment, from either kind of base model.

An Atelier checkpoint becomes a MiniEmbedder (atelier_mini.embed). A HuggingFace model
(an encoder such as all-MiniLM-L6-v2, or a causal LM) becomes an HFEmbedder with the
same interface — forward(texts), encode(texts), .model, .proj, .layer, .dim,
.max_length — so train_contrastive trains either one unchanged."""
from typing import Callable, Optional

import numpy as np
import torch


class HFEmbedder(torch.nn.Module):
    """Mean-pool one layer of a HuggingFace model over the attention mask, project,
    normalise. Layer -1 is the last hidden state; layer L the output of L blocks."""

    def __init__(self, path: str, device: str, layer: int = -1, dim: Optional[int] = None, max_length: int = 128):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True).to(device)
        self.layer = layer
        self.max_length = max_length
        width = int(self.model.config.hidden_size)
        self.n_blocks = int(getattr(self.model.config, "num_hidden_layers", 0) or 0)
        self.width = width
        self.dim = dim or width
        self.proj = torch.nn.Linear(width, self.dim, bias=False).to(device) if dim and dim != width else torch.nn.Identity()
        if isinstance(self.proj, torch.nn.Linear):
            torch.nn.init.normal_(self.proj.weight, 0, width**-0.5)

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(device)
        if self.layer < 0 or self.layer >= self.n_blocks:
            h = self.model(**enc).last_hidden_state
        else:
            h = self.model(**enc, output_hidden_states=True).hidden_states[self.layer]
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return torch.nn.functional.normalize(self.proj(pooled.float()), dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
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


def make_embedder(fmt: str, model_path: str, tokenizer_path: Optional[str], device: str, layer: int = -1, dim: Optional[int] = None, max_length: int = 128):
    """An untrained embedder on the base model: MiniEmbedder or HFEmbedder."""
    if fmt == "hf":
        return HFEmbedder(model_path, device, layer=layer, dim=dim, max_length=max_length)
    from atelier_mini.embed import MiniEmbedder
    from atelier_mini.model import MiniLM
    from atelier_mini.tok import MiniTokenizer

    model, _ = MiniLM.load(model_path, device)
    tok = MiniTokenizer.load(tokenizer_path)
    return MiniEmbedder(model, tok, layer=layer if 0 <= layer < len(model.blocks) else -1, dim=dim, max_length=max_length).to(device)


def n_blocks(emb) -> int:
    return emb.n_blocks if isinstance(emb, HFEmbedder) else len(emb.model.blocks)


def width(emb) -> int:
    return emb.width if isinstance(emb, HFEmbedder) else emb.model.config.n_embd


def save_embedder(emb, path, extra: dict) -> None:
    torch.save({"model_state": emb.model.state_dict(), "config": emb.model.config.to_dict(), "proj": emb.proj.state_dict() if isinstance(emb.proj, torch.nn.Linear) else None, "layer": emb.layer, "dim": emb.dim, "max_length": emb.max_length, "format": "hf" if isinstance(emb, HFEmbedder) else "atelier", **extra}, path)


def load_embedder(path, device: str, tokenizer_fallback: Optional[str] = None):
    """The embedder step 3 saved, of either kind."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("format") == "hf":
        emb = HFEmbedder(ck["base_model"], device, layer=ck["layer"], dim=ck["dim"] if ck["proj"] else None, max_length=ck["max_length"])
        emb.model.load_state_dict(ck["model_state"])
    else:
        from atelier_mini.embed import MiniEmbedder
        from atelier_mini.model import MiniConfig, MiniLM
        from atelier_mini.tok import MiniTokenizer

        model = MiniLM(MiniConfig(**ck["config"])).to(device)
        model.load_state_dict(ck["model_state"])
        tok = MiniTokenizer.load(ck.get("tokenizer") or tokenizer_fallback)
        emb = MiniEmbedder(model, tok, layer=ck["layer"], dim=ck["dim"] if ck["dim"] != model.config.n_embd else None, max_length=ck["max_length"]).to(device)
    if ck.get("proj") and isinstance(emb.proj, torch.nn.Linear):
        emb.proj.load_state_dict(ck["proj"])
    return emb, ck
