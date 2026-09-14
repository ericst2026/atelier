"""Step 4 — a direction, added, and what it does."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import Steering, difference_vector  # noqa: E402

parse_args()
P = params({"contrast": "working", "layer": -1, "strengths": ["0", "1", "2", "4", "8"], "n_test": 60, "sweep_layers": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
model, ck = MiniLM.load(I["model"], device)
model.eval()
tok = MiniTokenizer.load(I["tokenizer"])
world = World(lang=I.get("lang", "en"), seed=3)
system = I.get("system") or ck.get("system")
n_layers = len(model.blocks)
layer = int(P["layer"])
if layer < 0:
    layer = n_layers // 2

# the two sets of prompts are generated, so the effect can be counted rather than judged
pool = list(world.instructions(400, with_steps=True, seed=5150))
if P["contrast"] == "working":
    positive = ["\n".join(x["steps"]) + f"\n{world.answer_prefix} {x['answer']}" for x in pool[:80]]
    negative = [f"{world.answer_prefix} {x['answer']}" for x in pool[:80]]

    def measure(texts):
        return sum(1 for t in texts if len([l for l in t.strip().splitlines() if l.strip()]) > 1) / max(len(texts), 1)

    label = "answers that show working"
elif P["contrast"] == "brevity":
    short = sorted(pool, key=lambda x: len(x["target"]))[:80]
    long = sorted(pool, key=lambda x: -len(x["target"]))[:80]
    positive = [x["target"] for x in short]
    negative = [x["target"] for x in long]

    def measure(texts):
        return 1.0 - min(1.0, sum(len(t) for t in texts) / max(len(texts), 1) / 400)

    label = "brevity"
else:
    fam = pool[0]["family"]
    positive = [x["target"] for x in pool if x["family"] == fam][:80] or [pool[0]["target"]]
    negative = [x["target"] for x in pool if x["family"] != fam][:80]

    def measure(texts):
        return sum(1 for t in texts if any(c.isdigit() for c in t)) / max(len(texts), 1)

    label = f"the style of {fam}"

progress(10, f"building the direction at layer {layer}")
direction = difference_vector(model, tok, positive, negative, layer)
norm = float(np.linalg.norm(direction))
tasks = world.eval_set(int(P["n_test"]), seed=404_404)
prompts = [t["prompt"] for t in tasks]
strengths = sorted(float(s) for s in P["strengths"])

rows, samples = [], []
for si, s in enumerate(strengths):
    if s == 0:
        outs = [g[0] for g in generate(model, tok, prompts, 160, 0.0, batch_size=32, system=system)]
    else:
        with Steering(model, layer, direction, s):
            outs = [g[0] for g in generate(model, tok, prompts, 160, 0.0, batch_size=32, system=system)]
    words = [t.split() for t in outs]
    rows.append({
        "strength": s,
        "effect": measure(outs),
        "accuracy": sum(world.grade(o, t["answer"]) for o, t in zip(outs, tasks)) / len(tasks),
        "repetition": sum(1 - len(set(w)) / max(len(w), 1) for w in words) / max(len(words), 1),
        "chars": sum(len(o) for o in outs) / max(len(outs), 1),
    })
    samples += [{"strength": s, "question": t["prompt"][:90], "answer": o[:200]} for t, o in list(zip(tasks, outs))[:3]]
    progress(15 + 55 * (si + 1) / len(strengths), f"strength {s}: {label} {rows[-1]['effect']:.0%}, accuracy {rows[-1]['accuracy']:.0%}")

base = rows[0]
usable = [r for r in rows if r["strength"] > 0 and r["repetition"] < base["repetition"] + 0.25]
best = max(usable, key=lambda r: r["effect"] - base["effect"]) if usable else base
layer_rows = []
if bool(P["sweep_layers"]):
    probe_strength = best["strength"] or strengths[-1]
    for li in range(0, n_layers, max(1, n_layers // 6)):
        d = difference_vector(model, tok, positive, negative, li)
        with Steering(model, li, d, probe_strength):
            outs = [g[0] for g in generate(model, tok, prompts[:24], 160, 0.0, batch_size=24, system=system)]
        words = [t.split() for t in outs]
        layer_rows.append({"layer": li, "effect": measure(outs), "repetition": sum(1 - len(set(w)) / max(len(w), 1) for w in words) / max(len(words), 1)})
        progress(70 + 25 * len(layer_rows) / max(n_layers // max(1, n_layers // 6), 1), f"layer {li}: {layer_rows[-1]['effect']:.0%}")
(run_dir / "steer.json").write_text(json.dumps({"layer": layer, "norm": norm, "contrast": P["contrast"], "rows": rows, "layers": layer_rows}, indent=2))

R = Result()
R.metric("effect", f"Change in {label}", best["effect"] - base["effect"], "pct", "kept", help=f"strength {best['strength']} at layer {layer}; unsteered {base['effect']:.0%}")
R.metric("usable", "Strengths that kept the text readable", len(usable) / max(len(strengths) - 1, 1), "pct", "sky")
R.metric("accuracy_cost", "Accuracy change", best["accuracy"] - base["accuracy"], "pct", "dup" if best["accuracy"] < base["accuracy"] - 0.05 else "hold", help=f"unsteered {base['accuracy']:.0%}")
R.metric("norm", "Length of the direction", norm, "num", "raw")
R.chart("strength", "Effect, accuracy and damage against strength", rows, "strength", [{"key": "effect", "label": label, "color": "kept"}, {"key": "accuracy", "label": "Accuracy", "color": "sky"}, {"key": "repetition", "label": "Repetition", "color": "dup"}], "line", note="The useful window is where the first line has risen and the third has not. Past it the model repeats itself, which is technically steering.")
if layer_rows:
    R.chart("layers", "Which layer to inject at", layer_rows, "layer", [{"key": "effect", "label": label, "color": "kept"}, {"key": "repetition", "label": "Repetition", "color": "dup", "axis": "right"}], "line", note="Early layers change everything downstream and break the text; late layers are too close to the output to redirect it.")
R.table("samples", "What it wrote", [{"key": "strength", "label": "Strength"}, {"key": "question", "label": "Question"}, {"key": "answer", "label": "Answer"}], samples)
R.note("This is the test the probing step cannot do on its own. A direction found by a probe might be information the model ignores; if adding it changes the output in the predicted way, it is information the model uses.")
R.artifact(run_dir / "steer.json", "steer.json")
R.output("steer", str(run_dir / "steer.json")).output("effect", best["effect"] - base["effect"]).output("model", I["model"]).output("tokenizer", I["tokenizer"])
R.save()
