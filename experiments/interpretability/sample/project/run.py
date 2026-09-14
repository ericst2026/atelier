"""Try both locally.  python project/run.py --model pythia-160m"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "interpretability"))
from atelier_nlp import hf  # noqa: E402
from atelier_world import World  # noqa: E402
from interp import induction_heads, predict_probe, train_probe  # noqa: E402
from lib_interp import hidden_states  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="pythia-160m")
ap.add_argument("--n", type=int, default=600)
args = ap.parse_args()

mp = hf.model_path(args.model)
tok, model = hf.load_tokenizer(mp), hf.load_model(mp, dtype="fp32")
model.eval()
world = World(lang="en", seed=1)
tasks = world.eval_set(args.n, seed=246_801)
fams = sorted({t["family"] for t in tasks})
y = np.array([fams.index(t["family"]) for t in tasks])
H = hidden_states(model, tok, [t["prompt"] for t in tasks], "last", batch_size=16)
mid = H.shape[1] // 2
cut = int(len(y) * 0.75)
probe = train_probe(H[:cut, mid, :], y[:cut])
acc = float((predict_probe(probe, H[cut:, mid, :]) == y[cut:]).mean())
print(f"probe accuracy at layer {mid}: {acc:.1%} (guessing the commonest label gives {np.bincount(y[cut:]).max() / len(y[cut:]):.1%})")
print("induction heads:", induction_heads(model, tok))
