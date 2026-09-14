"""Your model. build_model(config) -> module with forward(idx, targets=None) -> (logits, loss)."""
import torch.nn as nn

from atelier_mini.model import MiniConfig, MiniLM


def build_model(config: dict) -> nn.Module:
    # The reference implementation is in atelier_mini/model.py — read it, copy the
    # parts you want to change into this file, and diverge.
    return MiniLM(MiniConfig(**config))
