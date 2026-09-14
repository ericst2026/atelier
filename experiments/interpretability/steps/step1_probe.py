"""Step 1 — which layer knows what."""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import fit_probe, hidden_states, probe_accuracy  # noqa: E402

parse_args()
P = params({"model": "pythia-160m", "target": ["family", "correctness"], "n_examples": 1200, "pooling": "last", "seed": 1})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
mp = hf.model_path(P["model"])
tok = hf.load_tokenizer(mp)
model = hf.load_model(mp)
model.eval()
world = World(lang="en", seed=int(P["seed"]))
n = int(P["n_examples"])
tasks = world.eval_set(n, seed=246_801)
prompts = [t["prompt"] for t in tasks]

progress(5, f"reading hidden states from {len(prompts)} prompts")
H = hidden_states(model, tok, prompts, P["pooling"], batch_size=16, progress=lambda d, t: progress(5 + 35 * d / t, f"{d}/{t}"))
n_layers = H.shape[1]

labels = {}
if "family" in P["target"]:
    fams = sorted({t["family"] for t in tasks})
    labels["which kind of question"] = (np.array([fams.index(t["family"]) for t in tasks]), len(fams))
if "correctness" in P["target"]:
    progress(45, "asking the model, to label whether it gets each one right")
    gens = hf.generate_batch(model, tok, [hf.chat_prompt(tok, t["prompt"]) for t in tasks], 96, 0.0, batch_size=16, progress=lambda d, t_: progress(45 + 25 * d / t_, f"{d}/{t_}"))
    correct = np.array([1 if world.grade(g[0], t["answer"]) else 0 for g, t in zip(gens, tasks)])
    if 0 < correct.sum() < len(correct):
        labels["whether it will be right"] = (correct, 2)
    else:
        print(f"[probe] skipping the correctness probe: the model got {correct.sum()}/{len(correct)} right, so there is nothing to separate", flush=True)
if "length" in P["target"]:
    lens = np.array([len(t["answer"]) for t in tasks])
    labels["how long the answer is"] = ((lens > np.median(lens)).astype(int), 2)
if not labels:
    raise SystemExit("No usable probe target — pick another, or a model that gets some questions right and some wrong.")

rng = random.Random(int(P["seed"]))
idx = list(range(len(prompts)))
rng.shuffle(idx)
cut = int(len(idx) * 0.75)
train_i, test_i = np.array(idx[:cut]), np.array(idx[cut:])

curves, best = {}, {}
for li, (name, (y, classes)) in enumerate(labels.items()):
    accs = []
    majority = float(max(np.bincount(y[test_i])) / len(test_i))
    for layer in range(n_layers):
        probe = fit_probe(H[train_i, layer, :], y[train_i], classes, seed=int(P["seed"]))
        accs.append({"layer": layer, name: probe_accuracy(probe, H[test_i, layer, :], y[test_i])})
        progress(70 + 28 * (li + (layer + 1) / n_layers) / len(labels), f"{name} · layer {layer}: {accs[-1][name]:.1%}")
    curves[name] = accs
    top = max(accs, key=lambda r: r[name])
    best[name] = {"layer": top["layer"], "accuracy": top[name], "majority": majority}

merged = {}
for name, rows in curves.items():
    for r in rows:
        merged.setdefault(r["layer"], {"layer": r["layer"]}).update(r)
(run_dir / "probe.json").write_text(json.dumps({"best": best, "curves": curves, "layers": n_layers, "model": P["model"]}, indent=2))

first = list(best)[0]
R = Result()
R.metric("best_layer", "Best layer", best[first]["layer"], "int", "kept", help=f"of {n_layers - 1}, probing for {first}")
R.metric("probe_accuracy", "Probe accuracy there", best[first]["accuracy"], "pct", "kept", help=f"always guessing the most common label gives {best[first]['majority']:.1%}")
for name, b in list(best.items())[1:]:
    R.metric(f"acc_{name[:12]}", f"Best for '{name}'", b["accuracy"], "pct", "sky", help=f"layer {b['layer']}, majority baseline {b['majority']:.1%}")
R.metric("layers", "Layers", n_layers - 1, "int", "hold", help=P["model"])
R.chart("curves", "Probe accuracy by layer", [merged[k] for k in sorted(merged)], "layer", [{"key": name, "label": name} for name in curves], "line", y_domain=[0, 1], note="A property readable from the embedding layer was in the input all along. One that only becomes readable in the middle is something the model computed.")
R.table("best", "Where each property is most readable", [{"key": "property", "label": "Property"}, {"key": "layer", "label": "Layer"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "majority", "label": "Guessing the common label", "fmt": "pct"}], [{"property": k, **v} for k, v in best.items()])
R.note("A linear probe finding a property does not prove the model uses it — the information can be present and ignored. It is evidence about representation, not about mechanism, and the steering step is the cheapest way to test whether a direction actually does anything.")
R.artifact(run_dir / "probe.json", "probe.json")
R.output("probe", str(run_dir / "probe.json")).output("model", P["model"]).output("layers", n_layers - 1).output("probe_accuracy", best[first]["accuracy"])
R.save()
