"""Low-rank adaptation, written out.

A frozen weight W gets a trainable correction BA, where A is r×in and B is out×r.
Train r(in+out) numbers instead of in·out, then either keep the adapter separate or
fold W + (α/r)·BA back into the original weight and ship one model."""
import math
from typing import Iterable

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("qkv", "proj", "gate", "up", "down")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B stays zero, so the model starts unchanged
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.drop(x) @ self.A.t() @ self.B.t() * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight.data + (self.B @ self.A) * self.scaling


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.0, targets: Iterable[str] = DEFAULT_TARGETS) -> dict:
    """Freeze everything, then replace the named Linears. Returns a parameter census."""
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t == child_name for t in targets):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout).to(child.weight.device))
                replaced.append({"module": full, "in": child.in_features, "out": child.out_features, "trainable": r * (child.in_features + child.out_features)})
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"modules": replaced, "total": total, "trainable": trainable, "share": trainable / max(total, 1), "r": r, "alpha": alpha}


def merge_lora(model) -> int:
    """Fold every adapter into its frozen weight and put the plain Linears back."""
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                base.weight.data = child.merged_weight()
                base.weight.requires_grad_(True)
                setattr(module, child_name, base)
                merged += 1
    for p in model.parameters():
        p.requires_grad_(True)
    return merged


def adapter_state(model) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad and (".A" in n or ".B" in n)}


def adapter_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for n, p in model.named_parameters() if p.requires_grad)
