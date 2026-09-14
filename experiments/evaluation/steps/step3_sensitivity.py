"""Step 3 — how much of the number is the harness rather than the model."""
import json
import os
import random
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_score import option_logprobs, pick  # noqa: E402
from lib_tasks import CHOICE, few_shot_prefix  # noqa: E402

parse_args()
P = params({"models": ["SmolLM2-135M", "SmolLM2-360M"], "shot_values": ["0", "2", "5"], "shuffle_options": True, "limit": 200})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
task_dir = Path(I["task_dir"])
choice_benchmarks = []
for name in I["benchmarks"]:
    items = read_jsonl(task_dir / f"{name}.jsonl", limit=int(P["limit"]))
    if items and items[0]["kind"] == CHOICE:
        choice_benchmarks.append((name, items))
if not choice_benchmarks:
    raise SystemExit("This step needs at least one multiple-choice benchmark from step 1.")

shots_list = sorted(int(s) for s in P["shot_values"])
configs = [{"name": f"{s}-shot", "shots": s, "shuffle": False} for s in shots_list]
if bool(P["shuffle_options"]):
    configs.append({"name": f"{shots_list[0]}-shot, options shuffled", "shots": shots_list[0], "shuffle": True})

rows, total = [], len(P["models"]) * len(configs) * len(choice_benchmarks)
done = 0
for model_name in P["models"]:
    mp = hf.model_path(model_name)
    tok = hf.load_tokenizer(mp, padding_side="right")
    model = hf.load_model(mp)
    for cfg in configs:
        accs = []
        for name, items in choice_benchmarks:
            prefix = few_shot_prefix(items, cfg["shots"])
            rng = random.Random(5)
            correct = 0
            for x in items:
                choices, answer = list(x["choices"]), x["answer"]
                if cfg["shuffle"]:
                    order = list(range(len(choices)))
                    rng.shuffle(order)
                    choices = [x["choices"][i] for i in order]
                    answer = order.index(x["answer"])
                scores = option_logprobs(model, tok, prefix + x["question"], choices, 8)
                correct += pick(scores, "mean") == answer
            accs.append({"benchmark": name, "accuracy": correct / len(items)})
            done += 1
            progress(5 + 90 * done / total, f"{model_name} · {cfg['name']} · {name}: {correct / len(items):.1%}")
        rows.append({"model": model_name, "config": cfg["name"], "accuracy": sum(a["accuracy"] for a in accs) / len(accs), "per_benchmark": {a["benchmark"]: a["accuracy"] for a in accs}})
    del model
    torch.cuda.empty_cache()

by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r)
spreads = {m: max(x["accuracy"] for x in v) - min(x["accuracy"] for x in v) for m, v in by_model.items()}
worst_spread = max(spreads.values())
# does the harness ever change which model looks better?
order_changes = 0
if len(P["models"]) > 1:
    for cfg in configs:
        ranking = sorted([r for r in rows if r["config"] == cfg["name"]], key=lambda r: -r["accuracy"])
        if ranking and ranking[0]["model"] != sorted([r for r in rows if r["config"] == configs[0]["name"]], key=lambda r: -r["accuracy"])[0]["model"]:
            order_changes += 1
(run_dir / "sensitivity.json").write_text(json.dumps(rows, indent=2))

R = Result()
R.metric("spread", "Largest swing from harness choices alone", worst_spread, "pct", "dup" if worst_spread > 0.05 else "kept", help="same model, same items, different defensible settings")
R.metric("order_changes", "Configurations that reorder the models", order_changes, "int", "dup" if order_changes else "kept", help=f"of {len(configs)}")
R.metric("best", "Best single number", max(r["accuracy"] for r in rows), "pct", "sky", help="the one you would quote if you were selling something")
R.metric("worst", "Worst", min(r["accuracy"] for r in rows), "pct", "hold")
R.chart("configs", "Accuracy under each configuration", [dict({"config": cfg["name"]}, **{r["model"]: r["accuracy"] for r in rows if r["config"] == cfg["name"]}) for cfg in configs], "config", [{"key": m, "label": m} for m in P["models"]], "bar", y_domain=[0, 1], note="If the bars change order between groups, the benchmark is not measuring what the leaderboard claims.")
R.chart("spread", "Swing per model", [{"model": m, "spread": s} for m, s in spreads.items()], "model", [{"key": "spread", "label": "Best minus worst", "color": "dup"}], "bar")
R.table("rows", "Every configuration", [{"key": "model", "label": "Model"}, {"key": "config", "label": "Harness"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}], rows)
R.artifact(run_dir / "sensitivity.json", "sensitivity.json")
R.output("sensitivity", str(run_dir / "sensitivity.json")).output("spread", worst_spread)
for k in ("task_dir", "benchmarks", "scores", "model", "limit"):
    if k in I:
        R.output(k, I[k])
R.save()
