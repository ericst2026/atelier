"""Step 4 — a direction, added, and what it costs."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import Steering, difference_vector, hidden_states  # noqa: E402

parse_args()
P = params({"positive": "", "negative": "", "layer": 6, "strengths": ["0", "1", "2", "4", "8"], "test_prompts": "", "sweep_layers": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
mp = hf.model_path(I["model"])
tok = hf.load_tokenizer(mp)
model = hf.load_model(mp, dtype="fp32")
model.eval()
n_layers = int(I.get("layers") or model.config.num_hidden_layers)
pos = [l.strip() for l in str(P["positive"]).splitlines() if l.strip()]
neg = [l.strip() for l in str(P["negative"]).splitlines() if l.strip()]
tests = [l.strip() for l in str(P["test_prompts"]).splitlines() if l.strip()]
if len(pos) < 2 or len(neg) < 2 or not tests:
    raise SystemExit("Give at least two prompts on each side and one prompt to generate from.")
layer = max(0, min(int(P["layer"]), n_layers - 1))


def score(texts: list[str]) -> dict:
    """How far the generations moved, and whether they are still English.
    The direction is scored by the probe's own separating vector, not by a judge."""
    H = hidden_states(model, tok, texts, "mean", max_length=64)[:, layer, :]
    proj = (H @ direction) / (np.linalg.norm(direction) + 1e-9)
    words = [t.split() for t in texts]
    repetition = sum(1 - len(set(w)) / max(len(w), 1) for w in words) / max(len(words), 1)
    empty = sum(1 for t in texts if len(t.strip()) < 5) / max(len(texts), 1)
    return {"projection": float(proj.mean()), "repetition": float(repetition), "empty": float(empty), "chars": sum(len(t) for t in texts) / max(len(texts), 1)}


progress(10, f"building the direction at layer {layer}")
direction = difference_vector(model, tok, pos, neg, layer)
norm = float(np.linalg.norm(direction))

strengths = sorted(float(s) for s in P["strengths"])
rows, samples = [], []
for si, s in enumerate(strengths):
    if s == 0:
        outs = [g[0] for g in hf.generate_batch(model, tok, tests, 48, 0.0, batch_size=8)]
    else:
        with Steering(model, layer, direction / max(norm, 1e-9) * norm, s):
            outs = [g[0] for g in hf.generate_batch(model, tok, tests, 48, 0.0, batch_size=8)]
    m = score(outs)
    rows.append({"strength": s, **m})
    samples += [{"strength": s, "prompt": p[:70], "output": o[:220]} for p, o in zip(tests, outs)]
    progress(15 + 45 * (si + 1) / len(strengths), f"strength {s}: projection {m['projection']:.2f}, repetition {m['repetition']:.0%}")

base = rows[0]
for r in rows:
    r["effect"] = r["projection"] - base["projection"]
    r["usable"] = 1.0 if (r["repetition"] < base["repetition"] + 0.25 and r["empty"] <= base["empty"] + 0.05) else 0.0
usable = [r for r in rows if r["usable"] and r["strength"] > 0]
best = max(usable, key=lambda r: r["effect"]) if usable else rows[0]

layer_rows = []
if bool(P["sweep_layers"]):
    probe_strength = best["strength"] if best["strength"] > 0 else strengths[-1]
    for li in range(0, n_layers, max(1, n_layers // 10)):
        d = difference_vector(model, tok, pos, neg, li)
        with Steering(model, li, d, probe_strength):
            outs = [g[0] for g in hf.generate_batch(model, tok, tests, 48, 0.0, batch_size=8)]
        H = hidden_states(model, tok, outs, "mean", max_length=64)[:, layer, :]
        proj = float(((H @ direction) / (np.linalg.norm(direction) + 1e-9)).mean())
        words = [t.split() for t in outs]
        rep = sum(1 - len(set(w)) / max(len(w), 1) for w in words) / max(len(words), 1)
        layer_rows.append({"layer": li, "effect": proj - base["projection"], "repetition": rep})
        progress(60 + 35 * (len(layer_rows)) / max(n_layers // max(1, n_layers // 10), 1), f"layer {li}: effect {proj - base['projection']:+.2f}")

(run_dir / "steer.json").write_text(json.dumps({"layer": layer, "norm": norm, "rows": rows, "layers": layer_rows}, indent=2))

R = Result()
R.metric("effect", "Effect at the best usable strength", best["effect"], "num", "kept", help=f"strength {best['strength']}, layer {layer}")
R.metric("usable", "Strengths that kept the text readable", sum(r["usable"] for r in rows if r["strength"] > 0) / max(len(strengths) - 1, 1), "pct", "sky")
R.metric("norm", "Length of the direction", norm, "num", "hold", help="the mean difference between the two sets of prompts")
R.metric("breakdown", "Repetition at the highest strength", rows[-1]["repetition"], "pct", "dup", help=f"unsteered: {base['repetition']:.0%}")
R.chart("strength", "Effect and damage against strength", rows, "strength", [{"key": "effect", "label": "Movement along the direction", "color": "kept"}, {"key": "repetition", "label": "Repetition", "color": "dup", "axis": "right"}], "line", note="The useful window is where the first line has risen and the second has not. Past it the model says the same word forever, which is technically steering.")
if layer_rows:
    R.chart("layers", "Which layer to inject at", layer_rows, "layer", [{"key": "effect", "label": "Effect", "color": "kept"}, {"key": "repetition", "label": "Repetition", "color": "dup", "axis": "right"}], "line", note="Early layers change everything downstream and break the text; late layers are too close to the output to redirect it. The middle is where this works.")
R.table("samples", "What it wrote", [{"key": "strength", "label": "Strength"}, {"key": "prompt", "label": "Prompt"}, {"key": "output", "label": "Continuation"}], samples)
R.note("Steering is the cheapest test of whether a direction found by probing means anything. If adding it changes the output in the expected way, the direction is used by the model and not merely present in it.")
R.artifact(run_dir / "steer.json", "steer.json")
R.output("steer", str(run_dir / "steer.json")).output("effect", best["effect"]).output("model", I["model"])
R.save()
