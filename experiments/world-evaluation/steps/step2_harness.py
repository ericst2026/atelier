"""Step 2 — score a model, three defensible ways."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_harness import option_logprobs, pick  # noqa: E402
from lib_items import few_shot_prefix  # noqa: E402

parse_args()
P = params({"scoring": ["sum", "mean", "unconditional"], "shots": 0, "max_new_tokens": 160, "batch_size": 16})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("model_run_run")
if not ref:
    raise SystemExit("Choose a model: a Pretraining step 3 run, or a Fine-tuning step 3 run.")
o = ref["outputs"]
model_path = o.get("sft_model") or o.get("model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, ck = MiniLM.load(model_path, device)
model.eval()
tok = MiniTokenizer.load(o["tokenizer"])
world = World(lang=I.get("lang", "en"), seed=1)
choice = read_jsonl(I["choice"])
open_items = read_jsonl(I["open"])
methods = [m for m in P["scoring"] if m in ("sum", "mean", "unconditional")] or ["mean"]
prefix = few_shot_prefix(choice, int(P["shots"]))

picks = {m: [] for m in methods}
for i, item in enumerate(choice):
    scores = option_logprobs(model, tok, prefix + item["question"], item["options"], int(P["batch_size"]))
    for m in methods:
        picks[m].append(pick(scores, m))
    if i % 25 == 0:
        progress(5 + 65 * i / len(choice), f"scoring item {i}/{len(choice)}")

n_opt = int(I.get("n_options", 4))
rows, by_family = [], {}
for m in methods:
    correct = [p == x["answer"] for p, x in zip(picks[m], choice)]
    rows.append({"method": m, "accuracy": sum(correct) / len(choice), "chance": 1 / n_opt})
    for c, x in zip(correct, choice):
        d = by_family.setdefault(x["family"], {})
        e = d.setdefault(m, [0, 0])
        e[0] += c
        e[1] += 1

open_acc = None
if open_items:
    progress(75, f"generating answers for {len(open_items)} open questions")
    system = o.get("system") or ck.get("system")
    gens = generate(model, tok, [x["question"] for x in open_items], int(P["max_new_tokens"]), 0.0, batch_size=32, system=system, progress=lambda d, t: progress(75 + 20 * d / t, f"{d}/{t}"))
    open_correct = [world.grade(g[0], x["answer"]) for g, x in zip(gens, open_items)]
    open_acc = sum(open_correct) / len(open_items)

best = max(rows, key=lambda r: r["accuracy"])
spread = max(r["accuracy"] for r in rows) - min(r["accuracy"] for r in rows)
(run_dir / "scores.json").write_text(json.dumps({"model": model_path, "shots": int(P["shots"]), "rows": rows, "open_accuracy": open_acc}, indent=2))

R = Result()
R.metric("accuracy", f"Accuracy · {best['method']}", best["accuracy"], "pct", "kept", help=f"chance is {1 / n_opt:.0%}")
R.metric("method_spread", "Spread across scoring rules", spread, "pct", "dup" if spread > 0.05 else "hold", help="the same model and the same items, scored three ways")
if open_acc is not None:
    R.metric("open_accuracy", "Open questions", open_acc, "pct", "sky", help="generated, then checked exactly — no options to choose between")
R.metric("params", "Parameters", model.num_params(), "int", "raw")
R.chart("methods", "Accuracy by scoring rule", [{"method": r["method"], "accuracy": r["accuracy"], "chance": r["chance"]} for r in rows], "method", [{"key": "accuracy", "label": "Correct", "color": "kept"}, {"key": "chance", "label": "Chance", "color": "hold"}], "bar", y_domain=[0, 1], note="A raw sum of log-probabilities favours short options; dividing by the token count favours long ones. Published numbers rarely say which was used.")
R.chart("families", "Accuracy by family", [dict({"family": f}, **{m: v[0] / v[1] for m, v in d.items()}) for f, d in sorted(by_family.items())], "family", [{"key": m, "label": m} for m in methods], "bar", y_domain=[0, 1])
R.table("rows", "Every number", [{"key": "method", "label": "Scoring"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "chance", "label": "Chance", "fmt": "pct"}], rows)
R.artifact(run_dir / "scores.json", "scores.json")
R.output("scores", str(run_dir / "scores.json")).output("model", model_path).output("tokenizer", o["tokenizer"]).output("accuracy", best["accuracy"]).output("best_method", best["method"])
for k in ("choice", "open", "lang", "seed", "n_options", "families"):
    if k in I:
        R.output(k, I[k])
R.save()
