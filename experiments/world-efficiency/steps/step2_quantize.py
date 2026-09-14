"""Step 2 — fewer bits, and the accuracy they cost."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.quant import model_bytes, quantize_model
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"bit_widths": ["8", "4"], "per_channel": True, "skip_head": True, "n_eval": 200})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=88)
tok = MiniTokenizer.load(I["tokenizer"])
tasks = world.eval_set(int(P["n_eval"]), seed=333_999)
system = I.get("system") or world.system_prompt
skip = ("head",) if bool(P["skip_head"]) else ()


def accuracy(model):
    gens = generate(model, tok, [t["prompt"] for t in tasks], 160, 0.0, batch_size=32, system=system)
    return sum(world.grade(g[0], t["answer"]) for g, t in zip(gens, tasks)) / len(tasks)


base, _ = MiniLM.load(I["model"], device)
progress(5, "measuring the unquantised model")
base_acc = accuracy(base)
base_bytes = sum(p.numel() * p.element_size() for p in base.parameters())
del base
torch.cuda.empty_cache()

settings = []
for b in sorted((int(x) for x in P["bit_widths"]), reverse=True):
    settings.append({"bits": b, "per_channel": True, "label": f"int{b}, per channel"})
    if bool(P["per_channel"]):
        settings.append({"bits": b, "per_channel": False, "label": f"int{b}, per tensor"})

rows, errors = [], {}
for i, s in enumerate(settings):
    model, _ = MiniLM.load(I["model"], device)
    report = quantize_model(model, s["bits"], s["per_channel"], skip)
    acc = accuracy(model)
    mb = model_bytes(model, s["bits"])
    rel = [v["rel"] for v in report.values()]
    rows.append({"setting": s["label"], "bits": s["bits"], "per_channel": s["per_channel"], "accuracy": acc, "mb": mb["bytes"] / 1e6, "mean_error": sum(rel) / max(len(rel), 1), "worst_error": max(rel) if rel else 0})
    errors[s["label"]] = report
    progress(10 + 85 * (i + 1) / len(settings), f"{s['label']}: {acc:.1%}, {mb['bytes'] / 1e6:.0f} MB")
    del model
    torch.cuda.empty_cache()

(run_dir / "quantize.json").write_text(json.dumps({"base_accuracy": base_acc, "base_mb": base_bytes / 1e6, "rows": rows}, indent=2))
int8 = next((r for r in rows if r["bits"] == 8 and r["per_channel"]), rows[0])
worst_layers = sorted(errors[rows[-1]["setting"]].items(), key=lambda kv: -kv[1]["rel"])[:12] if errors else []

R = Result()
R.metric("int8_accuracy", "int8, per channel", int8["accuracy"], "pct", "kept", help=f"unquantised: {base_acc:.1%}")
R.metric("base_accuracy", "Unquantised", base_acc, "pct", "raw")
R.metric("compression", "Memory saved at int8", 1 - int8["mb"] / (base_bytes / 1e6), "pct", "sky")
R.metric("mean_error", "Mean weight error at int8", int8["mean_error"], "pct", "hold", help="relative norm of the difference, per layer")
R.chart("accuracy", "Accuracy by bit width", [{"setting": r["setting"], "accuracy": r["accuracy"], "base": base_acc} for r in rows], "setting", [{"key": "accuracy", "label": "Accuracy", "color": "kept"}, {"key": "base", "label": "Unquantised", "color": "raw"}], "bar", y_domain=[0, 1], note="Per-tensor scales share one scale across the whole matrix, so one large weight ruins the resolution for the rest. At int4 that is usually the difference between working and noise.")
R.chart("memory", "Memory against accuracy", rows, "setting", [{"key": "mb", "label": "Megabytes", "color": "hold"}, {"key": "accuracy", "label": "Accuracy", "color": "kept", "axis": "right"}], "bar")
if worst_layers:
    R.chart("layers", "Which layers suffer most", [{"layer": k.split(".")[-2] + "." + k.split(".")[-1], "error": v["rel"]} for k, v in worst_layers], "layer", [{"key": "error", "label": "Relative error", "color": "dup"}], "bar", note="Layers with a few very large weights quantise worst. They are the candidates for keeping in full precision.")
R.table("rows", "Settings", [{"key": "setting", "label": "Setting"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "mb", "label": "Megabytes", "fmt": "num"}, {"key": "mean_error", "label": "Mean error", "fmt": "pct"}, {"key": "worst_error", "label": "Worst layer", "fmt": "pct"}], rows)
R.artifact(run_dir / "quantize.json", "quantize.json")
R.output("quantize", str(run_dir / "quantize.json")).output("base_accuracy", base_acc).output("int8_accuracy", int8["accuracy"])
for k in ("model", "tokenizer", "lang", "system", "baseline", "tokens_per_sec"):
    if k in I:
        R.output(k, I[k])
R.save()
