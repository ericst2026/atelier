"""Your probe and your induction-head finder."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "world-interpretability"))
from lib_interp import fit_probe, induction_scores, probe_predict  # noqa: E402


def train_probe(hidden: np.ndarray, labels: np.ndarray):
    return fit_probe(hidden, labels, classes=int(labels.max()) + 1)


def predict_probe(probe, hidden: np.ndarray) -> np.ndarray:
    return probe_predict(probe, hidden)


def induction_heads(model, tok) -> list[tuple[int, int]]:
    scores = induction_scores(model, tok, seq_len=48, n_sequences=16)
    ranked = sorted(((l, h, scores[l, h]) for l in range(scores.shape[0]) for h in range(scores.shape[1])), key=lambda r: -r[2])
    return [(l, h) for l, h, _ in ranked[:5]]
