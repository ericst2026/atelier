"""Step 2 — rank against quality."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, curves, read_jsonl
from atelier_mini.gen import generate
from atelier_mini.lora import adapter_bytes, apply_lora
from atelier_mini.model import MiniLM
from atelier_mini.tok import load_tokenizer
from atelier_mini.train import sft
from atelier_world import World

parse_args()
P = params({"ranks": ["2", "8", "32", "128"], "examples": 8000, "epochs": 2.0, "lr": 1e-3, "batch_size": 24, "n_eval": 200})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.loads(Path(I["lora_config"]).read_text())
world = World(lang=I.get("lang", "en"), seed=15)
tok = load_tokenizer(I["tokenizer"])
if I.get("data_source") == "prepared":
    # written by step 1 from the prepared dataset: its training rows, and questions held out from them
    rows = read_jsonl(I["data_train"], limit=int(P["examples"]))
    tasks = read_jsonl(I["data_val"], limit=int(P["n_eval"]))
else:
    rows = list(world.instructions(int(P["examples"]), with_steps=True))
    tasks = world.eval_set(int(P["n_eval"]), seed=616_003)
system = world.system_prompt


def accuracy(model):
    gens = generate(model, tok, [t["prompt"] for t in tasks], 192, 0.0, batch_size=32, system=system)
    return sum(world.grade(g[0], t["answer"]) for g, t in zip(gens, tasks)) / len(tasks)


base, _ = MiniLM.load(I["base_model"], device)
progress(5, "measuring the base model")
base_acc = accuracy(base)
del base
torch.cuda.empty_cache()

ranks = sorted(int(r) for r in P["ranks"])
results = []
for i, r in enumerate(ranks):
    model, _ = MiniLM.load(I["base_model"], device)
    census = apply_lora(model, r, 2 * r, cfg["dropout"], tuple(cfg["targets"]))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    steps_total = max(1, int(len(rows) * float(P["epochs"]) / int(P["batch_size"])))

    def on_log(x, i=i, r=r, steps_total=steps_total):
        progress(10 + 80 * (i + x["step"] / steps_total) / len(ranks), f"rank {r} · step {x['step']}/{steps_total}", step=x["step"], **{f"loss_rank{r}": v for k, v in curves(x, "loss").items()})

    res = sft(model, tok, rows, run_dir / f"r{r}", None, epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), system=system, device=device, on_log=on_log)
    acc = accuracy(model)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    results.append({"rank": r, "trainable": census["trainable"], "share": census["share"], "accuracy": acc, "adapter_mb": adapter_bytes(model) / 1e6, "peak_gb": peak, "seconds": res["elapsed_sec"]})
    progress(10 + 85 * (i + 1) / len(ranks), f"rank {r}: {acc:.1%} with {census['trainable']:,} trainable parameters", step=r, accuracy=acc)
    del model
    torch.cuda.empty_cache()

best = max(results, key=lambda x: x["accuracy"])
efficient = max(results, key=lambda x: x["accuracy"] / max(x["trainable"], 1))
(run_dir / "sweep.json").write_text(json.dumps({"base_accuracy": base_acc, "results": results}, indent=2))

R = Result()
R.metric("best_rank", "Best rank", best["rank"], "int", "kept", help=f"{best['accuracy']:.1%}")
R.metric("best_accuracy", "Best accuracy", best["accuracy"], "pct", "kept", help=f"base model: {base_acc:.1%}")
R.metric("efficient_rank", "Best accuracy per trained weight", efficient["rank"], "int", "sky", help=f"{efficient['accuracy']:.1%} with {efficient['trainable']:,} parameters")
R.metric("adapter_mb", "Adapter size at the best rank", best["adapter_mb"], "num", "hold", help="megabytes you would ship per adapted task")
R.chart("quality", "Accuracy against rank", [{"rank": r["rank"], "accuracy": r["accuracy"], "base": base_acc} for r in results], "rank", [{"key": "accuracy", "label": "Accuracy", "color": "kept"}, {"key": "base", "label": "Before adapting", "color": "raw"}], "line", x_log=True, y_domain=[0, 1], note="Flat over a wide range, then a cliff. The cliff is the interesting part.")
R.chart("tradeoff", "Accuracy against trainable parameters", [{"trainable": r["trainable"], "accuracy": r["accuracy"]} for r in results], "trainable", [{"key": "accuracy", "label": "Accuracy", "color": "sky"}], "line", x_log=True, y_domain=[0, 1])
R.chart("cost", "Memory and time by rank", results, "rank", [{"key": "peak_gb", "label": "Peak GB", "color": "raw"}, {"key": "seconds", "label": "Seconds", "color": "hold", "axis": "right"}], "bar", note="Barely moves. LoRA's saving is in optimizer state, and at this model size the activations dominate anyway." + ("" if device == "cuda" else " This run was on CPU: peak GB is GPU memory and was not measured, so it reads 0."))
R.table("results", "The sweep", [{"key": "rank", "label": "Rank"}, {"key": "trainable", "label": "Trainable", "fmt": "int"}, {"key": "share", "label": "Share", "fmt": "pct"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "adapter_mb", "label": "Adapter MB", "fmt": "num"}, {"key": "seconds", "label": "Seconds", "fmt": "num"}], results)
R.artifact(run_dir / "sweep.json", "sweep.json")
R.output("sweep", str(run_dir / "sweep.json")).output("best_rank", best["rank"]).output("base_accuracy", base_acc)
for k in ("lora_config", "base_model", "tokenizer", "lang", "system", "model_format", "adapter", "model_label", "data_source", "data_label", "data_train", "data_val"):
    if k in I:
        R.output(k, I[k])
R.save()
