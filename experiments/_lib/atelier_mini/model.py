"""A decoder-only transformer with pre-norm blocks, RoPE and SwiGLU.

Deliberately close to what a modern small model looks like, and small enough to
read in one sitting. Sizes are chosen so a lesson fits in a lesson."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# name: (layers, heads, width, context) — parameter counts assume a 4k vocabulary
PRESETS = {
    "tiny": (6, 6, 288, 256),     # ~10M  · minutes on one A6000
    "small": (8, 8, 512, 512),    # ~30M  · half an hour
    "medium": (12, 12, 768, 512),  # ~90M  · a couple of hours
    "large": (16, 16, 1024, 1024),  # ~200M · overnight, or several GPUs
}


@dataclass
class MiniConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 288
    dropout: float = 0.0
    tie_weights: bool = True
    rope_base: float = 10000.0
    rope_scale: float = 1.0   # >1 stretches the positions: position interpolation

    @classmethod
    def preset(cls, name: str, vocab_size: int, **over) -> "MiniConfig":
        L, H, D, T = PRESETS[name]
        return cls(vocab_size=vocab_size, n_layer=L, n_head=H, n_embd=D, block_size=T, **over)

    def to_dict(self) -> dict:
        return asdict(self)


def rope_cache(head_dim: int, seq: int, device, base: float = 10000.0, scale: float = 1.0):
    """scale > 1 divides every position by scale — position interpolation, which keeps
    a model trained at length L usable at length L·scale without retraining from scratch."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float() / scale
    freqs = torch.outer(t, inv)
    return torch.cos(freqs)[None, None], torch.sin(freqs)[None, None]


def apply_rope(x, cos, sin):
    """Rotate pairs of channels by a position-dependent angle. Positions enter the
    model through the angle rather than through an added embedding."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    T = x.shape[-2]
    c, s = cos[..., :T, :], sin[..., :T, :]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: MiniConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head, self.head_dim = c.n_head, c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.dropout = c.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, c: MiniConfig):
        super().__init__()
        hidden = int(8 * c.n_embd / 3 / 64 + 0.5) * 64      # ≈ 8/3·d, rounded for the GPU
        self.gate = nn.Linear(c.n_embd, hidden, bias=False)
        self.up = nn.Linear(c.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, c.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: MiniConfig):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(c.n_embd), nn.RMSNorm(c.n_embd)
        self.attn, self.mlp = Attention(c), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class MiniLM(nn.Module):
    def __init__(self, c: MiniConfig):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.norm = nn.RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        if c.tie_weights:
            self.head.weight = self.embed.weight
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))
        self._cos = self._sin = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _rope(self, T: int, device):
        if self._cos is None or self._cos.shape[-2] < T or self._cos.device != device:
            self._cos, self._sin = rope_cache(self.config.n_embd // self.config.n_head, max(T, self.config.block_size), device, self.config.rope_base, self.config.rope_scale)
        return self._cos, self._sin

    def forward_cached(self, idx, cache=None):
        """One step with a key/value cache. cache is a list of (k, v) per block, or
        None on the first call. Returns (logits, new_cache)."""
        B, T = idx.shape
        past = cache[0][0].shape[-2] if cache else 0
        cos, sin = self._rope(past + T, idx.device)
        cos, sin = cos[..., past : past + T, :], sin[..., past : past + T, :]
        x = self.embed(idx)
        new_cache = []
        for i, block in enumerate(self.blocks):
            h = block.n1(x)
            Bq, Tq, C = h.shape
            q, k, v = block.attn.qkv(h).split(C, dim=2)
            nh, hd = block.attn.n_head, block.attn.head_dim
            q = q.view(Bq, Tq, nh, hd).transpose(1, 2)
            k = k.view(Bq, Tq, nh, hd).transpose(1, 2)
            v = v.view(Bq, Tq, nh, hd).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            if cache:
                pk, pv = cache[i]
                k, v = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
            new_cache.append((k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=Tq > 1)
            x = x + block.attn.proj(y.transpose(1, 2).contiguous().view(Bq, Tq, C))
            x = x + block.mlp(block.n2(x))
        return self.head(self.norm(x)), new_cache

    def forward(self, idx, targets=None, loss_mask=None, reduction: str = "mean"):
        """targets: next-token ids (-100 to ignore). loss_mask: 1 where the loss counts,
        which is how SFT trains on the answer without training on the question."""
        B, T = idx.shape
        cos, sin = self._rope(T, idx.device)
        x = self.drop(self.embed(idx))
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            flat = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction="none")
            if loss_mask is not None:
                m = loss_mask.reshape(-1).float()
                loss = (flat * m).sum() / m.sum().clamp(min=1) if reduction == "mean" else (flat * m).view(B, T)
            else:
                loss = flat.mean() if reduction == "mean" else flat.view(B, T)
        return logits, loss

    @torch.no_grad()
    def attention_maps(self, idx):
        """The attention matrices, one per layer: (layers, heads, T, T).

        The fast path uses fused attention, which never materialises these — so this
        recomputes them the slow, explicit way. Only for looking at, never for training.
        """
        B, T = idx.shape
        cos, sin = self._rope(T, idx.device)
        x = self.embed(idx)
        maps = []
        mask = torch.full((T, T), float("-inf"), device=idx.device).triu(1)
        for block in self.blocks:
            h = block.n1(x)
            q, k, v = block.attn.qkv(h).split(self.config.n_embd, dim=2)
            nh, hd = block.attn.n_head, block.attn.head_dim
            q = q.view(B, T, nh, hd).transpose(1, 2)
            k = k.view(B, T, nh, hd).transpose(1, 2)
            v = v.view(B, T, nh, hd).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd) + mask
            att = F.softmax(att.float(), dim=-1)
            maps.append(att)
            y = (att.to(v.dtype) @ v).transpose(1, 2).contiguous().view(B, T, self.config.n_embd)
            x = x + block.attn.proj(y)
            x = x + block.mlp(block.n2(x))
        return torch.stack(maps, dim=0)

    def hidden(self, idx, layer: int = -1):
        """The residual stream after `layer` blocks (-1 means the last, 0 the embedding).
        This is what turns a language model into an encoder."""
        B, T = idx.shape
        cos, sin = self._rope(T, idx.device)
        x = self.embed(idx)
        n = len(self.blocks) if layer < 0 else min(layer, len(self.blocks))
        for block in self.blocks[:n]:
            x = block(x, cos, sin)
        return self.norm(x) if n == len(self.blocks) else x

    def num_params(self, embeddings: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n if embeddings else n - self.embed.weight.numel()

    def optimizers(self, weight_decay: float, lr: float, betas=(0.9, 0.95)):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        plain = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": plain, "weight_decay": 0.0}], lr=lr, betas=betas, fused=torch.cuda.is_available())

    def save(self, path, extra: dict | None = None):
        torch.save({"model_state": self.state_dict(), "config": self.config.to_dict(), **(extra or {})}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = cls(MiniConfig(**ck["config"])).to(device)
        m.load_state_dict(ck["model_state"])
        return m, ck


def estimate(c: MiniConfig) -> dict:
    """Parameter and FLOP counts without allocating anything."""
    d, L, V, T = c.n_embd, c.n_layer, c.vocab_size, c.block_size
    hidden = int(8 * d / 3 / 64 + 0.5) * 64
    attn = L * 4 * d * d
    mlp = L * 3 * d * hidden
    norms = L * 2 * d + d
    emb = V * d * (1 if c.tie_weights else 2)
    total = attn + mlp + norms + emb
    non_emb = attn + mlp + norms
    return {
        "total": total, "non_embedding": non_emb, "attention": attn, "mlp": mlp,
        "norms": norms, "embedding": emb, "mlp_hidden": hidden,
        "flops_per_token": 6 * non_emb + 12 * L * d * T,
        "chinchilla_tokens": 20 * total,
    }
