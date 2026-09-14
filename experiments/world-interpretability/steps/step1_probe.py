"""Step 1 — which layer knows what, in a model you trained."""
import json
import os
import random
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
from lib_interp import fit_probe, hidden_states, probe_accuracy  # noqa: E402

parse_args()
P = params({"targets": ["family", "correctness"], "n_examples": 1500, "pooling": "last", "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("model_run_run")
if not ref:
    raise SystemExit("Choose a model: a Fine-tuning step 3 run, or a Pretraining step 3 run.")
o = ref["outputs"]
model_path = o.get("sft_model") or o.get("model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, ck = MiniLM.load(model_path, device)
model.eval()
tok = MiniTokenizer.load(o["tokenizer"])
world = World(lang=o.get("lang", "en"), seed=int(P["seed"]))
system = o.get("system") or ck.get("system")
tasks = world.eval_set(int(P["n_examples"]), seed=246_801)
prompts = [t["prompt"] for t in tasks]

progress(5, f"reading hidden states from {len(prompts)} prompts")
H = hidden_states(model, tok, prompts, P["pooling"], progress=lambda d, t: progress(5 + 35 * d / t, f"{d}/{t}"))
n_layers = H.shape[1]

labels = {}
if "family" in P["targets"]:
    fams = sorted({t["family"] for t in tasks})
    labels["which kind of question"] = (np.array([fams.index(t["family"]) for t in tasks]), len(fams))
if "difficulty" in P["targets"]:
    diffs = sorted({t["difficulty"] for t in tasks})
    labels["how hard it is"] = (np.array([diffs.index(t["difficulty"]) for t in tasks]), len(diffs))
if "correctness" in P["targets"]:
    progress(45, "asking the model, to label which ones it gets right")
    gens = generate(model, tok, prompts, 160, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(45 + 25 * d / t, f"{d}/{t}"))
    correct = np.array([1 if world.grade(g[0], t["answer"]) else 0 for g, t in zip(gens, tasks)])
    if 0 < correct.sum() < len(correct):
        labels["whether it will be right"] = (correct, 2)
    else:
        print(f"[probe] skipping the correctness probe: {correct.sum()}/{len(correct)} correct, so there is nothing to separate", flush=True)
if not labels:
    raise SystemExit("No usable probe target — pick another, or a model that gets some right and some wrong.")

rng = random.Random(int(P["seed"]))
idx = list(range(len(prompts)))
rng.shuffle(idx)
cut = int(len(idx) * 0.75)
train_i, test_i = np.array(idx[:cut]), np.array(idx[cut:])
curves, best = {}, {}
for li, (name, (y, classes)) in enumerate(labels.items()):
    accs, majority = [], float(max(np.bincount(y[test_i])) / len(test_i))
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
(run_dir / "probe.json").write_text(json.dumps({"best": best, "curves": curves, "layers": n_layers}, indent=2))
first = list(best)[0]

R = Result()
R.metric("best_layer", "Best layer", best[first]["layer"], "int", "kept", help=f"of {n_layers - 1}, probing for {first}")
R.metric("probe_accuracy", "Probe accuracy there", best[first]["accuracy"], "pct", "kept", help=f"guessing the commonest label gives {best[first]['majority']:.1%}")
for name, b in list(best.items())[1:]:
    R.metric(f"acc_{name[:10].replace(' ', '_')}", f"Best for '{name}'", b["accuracy"], "pct", "sky", help=f"layer {b['layer']}, baseline {b['majority']:.1%}")
R.metric("layers", "Layers", n_layers - 1, "int", "hold", help=f"{model.num_params():,} parameters")
R.chart("curves", "Probe accuracy by layer", [merged[k] for k in sorted(merged)], "layer", [{"key": name, "label": name} for name in curves], "line", y_domain=[0, 1], note="Readable at layer 0 means it was in the input. Readable only later means the model computed it.")
R.table("best", "Where each property is most readable", [{"key": "property", "label": "Property"}, {"key": "layer", "label": "Layer"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "majority", "label": "Commonest label", "fmt": "pct"}], [{"property": k, **v} for k, v in best.items()])
R.note("A linear probe finding a property does not prove the model uses it: the information can be present and ignored. That is what the steering step tests, and it is the reason these four techniques belong in one experiment rather than four.")
R.artifact(run_dir / "probe.json", "probe.json")
R.output("probe", str(run_dir / "probe.json")).output("model", model_path).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("system", system).output("layers", n_layers - 1).output("probe_accuracy", best[first]["accuracy"])
R.save()
