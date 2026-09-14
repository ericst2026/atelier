"""Your model. `build_model(config)` must return a module whose forward(idx, targets) -> (logits, loss)."""
import sys
from pathlib import Path

import torch.nn as nn

# The reference implementation lives in the experiment package; start from it and diverge.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pretrain"))
from lib.model import GPT, GPTConfig  # noqa: E402


def build_model(config: dict) -> nn.Module:
    return GPT(GPTConfig(**config))
