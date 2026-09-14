"""Step 3 — LoRA against a full fine-tune, same data and same steps."""
import json
import os
import time
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.lora import adapter_bytes, apply_lora
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import sft
from atelier_world import World

parse_args()
P = params({"r": 16, "examples": 8000, "epochs": 2.0, "lora_lr": 1e-3, "full_lr": 2e-4, "n_eval": 250})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.loads(Path(I["lora_config"]).read_text())
world = World(lang=I.get("lang", "en"), seed=15)
tok = MiniTokenizer.load(I["tokenizer"])
rows = list(world.instructions(int(P["examples"]), with_steps=True))
tasks = world.eval_set(int(P["n_eval"]), seed=616_003)
system = world.system_prompt
steps_total = max(1, int(len(rows) * float(P["epochs"]) / 24))


def accuracy(model):
    gens = generate(model, tok, [t["prompt"] for t in tasks], 192, 0.0, batch_size=32, system=system)
    correct = [world.grade(g[0], t["answer"]) for g, t in zip(gens, tasks)]
    by_family = {}
    for c, t in zip(correct, tasks):
        f = by_family.setdefault(t["family"], [0, 0])
        f[0] += c
        f[1] += 1
    return sum(correct) / len(tasks), {k: v[0] / v[1] for k, v in by_family.items()}


runs = []
for j, mode in enumerate(("lora", "full")):
    model, _ = MiniLM.load(I["base_model"], device)
    if mode == "lora":
        census = apply_lora(model, int(P["r"]), 2 * int(P["r"]), cfg["dropout"], tuple(cfg["targets"]))
        trainable, lr = census["trainable"], float(P["lora_lr"])
    else:
        trainable, lr = sum(p.numel() for p in model.parameters()), float(P["full_lr"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = sft(model, tok, rows, run_dir / mode, None, epochs=float(P["epochs"]), batch_size=24, lr=lr, system=system, device=device,
              on_log=lambda x, j=j, mode=mode: progress(5 + 45 * j + 35 * x["step"] / steps_total, f"{mode} · step {x['step']}/{steps_total}", step=x["step"], **{k: v for k, v in x.items() if k == "loss"}))
    acc, by_family = accuracy(model)
    runs.append({
        "mode": "LoRA" if mode == "lora" else "full fine-tune",
        "trainable": trainable, "accuracy": acc, "by_family": by_family,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
        "seconds": time.time() - t0,
        "ship_mb": (adapter_bytes(model) if mode == "lora" else sum(p.numel() * p.element_size() for p in model.parameters())) / 1e6,
        "checkpoint": res["checkpoint"],
    })
    if mode == "lora":
        model.save(run_dir / "lora_model.pt", {"lora": True, "r": int(P["r"]), "targets": cfg["targets"], "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["base_model"]})
    del model
    torch.cuda.empty_cache()

lora_run, full_run = runs[0], runs[1]
gap = full_run["accuracy"] - lora_run["accuracy"]
families = sorted(set(lora_run["by_family"]) | set(full_run["by_family"]))

R = Result()
R.metric("lora_accuracy", "LoRA", lora_run["accuracy"], "pct", "kept", help=f"{lora_run['trainable']:,} trainable parameters")
R.metric("full_accuracy", "Full fine-tune", full_run["accuracy"], "pct", "raw", help=f"{full_run['trainable']:,} trainable parameters")
R.metric("gap", "Gap", gap, "pct", "dup" if gap > 0.05 else "hold", help="full minus LoRA")
R.metric("ship_ratio", "Storage per adapted model", full_run["ship_mb"] / max(lora_run["ship_mb"], 1e-9), "num", "sky", help=f"{lora_run['ship_mb']:.1f} MB against {full_run['ship_mb']:.0f} MB")
R.chart("accuracy", "Accuracy", [{"mode": r["mode"], "accuracy": r["accuracy"]} for r in runs], "mode", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("cost", "What each costs", [{"mode": r["mode"], "peak_gb": r["peak_gb"], "seconds": r["seconds"]} for r in runs], "mode", [{"key": "peak_gb", "label": "Peak GB", "color": "raw"}, {"key": "seconds", "label": "Seconds", "color": "hold", "axis": "right"}], "bar")
R.chart("families", "Accuracy by family", [{"family": f, "LoRA": lora_run["by_family"].get(f, 0), "full": full_run["by_family"].get(f, 0)} for f in families], "family", [{"key": "LoRA", "label": "LoRA", "color": "kept"}, {"key": "full", "label": "Full", "color": "raw"}], "bar", y_domain=[0, 1], note="Where LoRA loses, it usually loses on the families that need the most change from the base model.")
R.table("runs", "Side by side", [{"key": "mode", "label": "Method"}, {"key": "trainable", "label": "Trainable", "fmt": "int"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "peak_gb", "label": "Peak GB", "fmt": "num"}, {"key": "seconds", "label": "Seconds", "fmt": "num"}, {"key": "ship_mb", "label": "MB to ship", "fmt": "num"}], runs)
R.artifact(run_dir / "lora_model.pt", "lora_model.pt")
R.output("lora_model", str(run_dir / "lora_model.pt")).output("full_model", full_run["checkpoint"]).output("lora_accuracy", lora_run["accuracy"]).output("full_accuracy", full_run["accuracy"]).output("r", int(P["r"]))
for k in ("lora_config", "base_model", "tokenizer", "lang"):
    R.output(k, I[k])
R.save()
