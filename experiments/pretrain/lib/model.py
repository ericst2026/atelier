"""A small GPT (decoder-only transformer), close to nanoGPT, written for reading."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False

    def to_dict(self):
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.n_head, self.n_embd, self.dropout = c.n_head, c.n_embd, c.dropout
        self.resid_drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.config = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.wpe.weight.numel() if non_embedding else n

    def forward(self, idx, targets=None, reduction: str = "mean"):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
        return logits, loss

    def configure_optimizers(self, weight_decay: float, lr: float, betas=(0.9, 0.95)):
        decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=betas, fused=torch.cuda.is_available())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.block_size :]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1) if temperature > 0 else probs.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
        return idx


def estimate(c: GPTConfig) -> dict:
    """Closed-form parameter and FLOP estimates (no allocation)."""
    d, L, V, T = c.n_embd, c.n_layer, c.vocab_size, c.block_size
    attn = L * (4 * d * d)
    mlp = L * (8 * d * d)
    ln = L * 2 * (2 * d) + 2 * d
    emb = V * d
    pos = T * d
    total = attn + mlp + ln + emb + pos
    non_emb = total - emb - pos
    flops_per_token = 6 * non_emb + 12 * L * d * T  # weights + attention scores
    return {"total": total, "non_embedding": non_emb, "attention": attn, "mlp": mlp, "layernorm": ln, "embedding": emb, "position": pos, "flops_per_token": flops_per_token, "chinchilla_tokens": 20 * non_emb}
