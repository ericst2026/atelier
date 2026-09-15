"""Step 3 — how much of the number is the harness rather than the model."""
import json
import os
import random
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_world.prepared import choose_model, load_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_harness import pick, score_options  # noqa: E402
from lib_items import few_shot_prefix  # noqa: E402

parse_args()
P = params({"compare_source": "generated", "compare_material": None, "shot_values": ["0", "2", "5"], "shuffle_options": True, "limit": 300})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
choice = read_jsonl(I["choice"], limit=int(P["limit"]))
models = [("this run", {"format": I.get("model_format", "atelier"), "model": I["model"], "tokenizer": I["tokenizer"], "adapter": I.get("adapter"), "system": I.get("system")})]
cmp = I.get("compare_run_run")
if str(P["compare_source"]) == "prepared" or cmp:
    # the second model is optional: a run of yours, or a prepared model
    run_key = "sft_model" if cmp and cmp["outputs"].get("sft_model") else "model"
    other = choose_model(P, I, run_key="compare_run", run_model_key=run_key, source_key="compare_source", material_key="compare_material", hint="Choose a second model to compare with, or leave it empty.")
    models.append((f"run {cmp['id']}" if other["source"] == "generated" else other["label"], other))

shots = sorted(int(s) for s in P["shot_values"])
configs = [{"name": f"{s}-shot", "shots": s, "shuffle": False} for s in shots]
if bool(P["shuffle_options"]):
    configs.append({"name": f"{shots[0]}-shot, shuffled", "shots": shots[0], "shuffle": True})

rows, total, done = [], len(models) * len(configs), 0
for label, info in models:
    lm = load_lm(info, device)
    lm.model.eval()
    for cfg in configs:
        prefix = few_shot_prefix(choice, cfg["shots"])
        rng = random.Random(5)
        correct = 0
        for item in choice:
            options, answer = list(item["options"]), item["answer"]
            if cfg["shuffle"]:
                order = list(range(len(options)))
                rng.shuffle(order)
                options = [item["options"][j] for j in order]
                answer = order.index(item["answer"])
            scores = score_options(lm, prefix + item["question"], options, 16)
            correct += pick(scores, "mean") == answer
        rows.append({"model": label, "config": cfg["name"], "accuracy": correct / len(choice)})
        done += 1
        progress(5 + 90 * done / total, f"{label} · {cfg['name']}: {correct / len(choice):.1%}")
    del lm
    if device == "cuda":
        torch.cuda.empty_cache()

by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r)
spreads = {m: max(x["accuracy"] for x in v) - min(x["accuracy"] for x in v) for m, v in by_model.items()}
worst = max(spreads.values())
order_changes = 0
if len(models) > 1:
    baseline = max((r for r in rows if r["config"] == configs[0]["name"]), key=lambda r: r["accuracy"])["model"]
    for cfg in configs[1:]:
        top = max((r for r in rows if r["config"] == cfg["name"]), key=lambda r: r["accuracy"])["model"]
        order_changes += top != baseline
(run_dir / "sensitivity.json").write_text(json.dumps(rows, indent=2))

R = Result()
R.metric("spread", "Largest swing from harness choices", worst, "pct", "dup" if worst > 0.05 else "kept")
R.metric("best", "Best number available", max(r["accuracy"] for r in rows), "pct", "sky", help="the one you would quote if you were selling something")
R.metric("worst", "Worst", min(r["accuracy"] for r in rows), "pct", "hold")
if len(models) > 1:
    R.metric("order_changes", "Configurations that reorder the models", order_changes, "int", "dup" if order_changes else "kept", help=f"of {len(configs) - 1} compared with the first")
R.chart("configs", "Accuracy under each configuration", [dict({"config": cfg["name"]}, **{m: next((r["accuracy"] for r in rows if r["config"] == cfg["name"] and r["model"] == m), None) for m, _ in models}) for cfg in configs], "config", [{"key": m, "label": m} for m, _ in models], "bar", y_domain=[0, 1], note="If the bars change order between groups, the benchmark is not measuring what a leaderboard would claim it measures.")
R.chart("spread", "Swing per model", [{"model": m, "spread": s} for m, s in spreads.items()], "model", [{"key": "spread", "label": "Best minus worst", "color": "dup"}], "bar")
R.table("rows", "Every configuration", [{"key": "model", "label": "Model"}, {"key": "config", "label": "Harness"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}], rows)
R.artifact(run_dir / "sensitivity.json", "sensitivity.json")
R.output("sensitivity", str(run_dir / "sensitivity.json")).output("spread", worst)
for k in ("choice", "open", "model", "tokenizer", "scores", "lang", "seed", "n_options", "accuracy", "model_format", "adapter", "model_label", "system", "data_source", "data_label"):
    if k in I:
        R.output(k, I[k])
R.save()
