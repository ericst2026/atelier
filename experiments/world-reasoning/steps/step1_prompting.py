"""Step 1 — what each way of asking is worth, and what it costs."""
import os
from collections import Counter
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_eval": 250, "strategies": ["direct", "cot", "fewshot", "self_consistency"], "k": 8, "temperature": 0.8, "max_new_tokens": 192})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("model_run_run")
if not ref:
    raise SystemExit("Choose a model: a Fine-tuning step 3 run, or a Pretraining step 3 run.")
o = ref["outputs"]
model_path = o.get("sft_model") or o.get("model")
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=o.get("lang", "en"), seed=3)
model, ck = MiniLM.load(model_path, device)
tok = MiniTokenizer.load(o["tokenizer"])
system = o.get("system") or ck.get("system") or world.system_prompt
tasks = world.eval_set(int(P["n_eval"]), seed=222_004)
shots = world.eval_set(2, seed=13)
fewshot = "\n\n".join(f"{s['prompt']}\n" + "\n".join(s["steps"]) + f"\n{world.answer_prefix} {s['answer']}" for s in shots)
DIRECT = world.pack["instruction_system"] + " " + ("Give only the final answer." if world.lang == "en" else "答えだけを書いてください。")


def majority(answers):
    c = Counter(a for a in answers if a)
    return c.most_common(1)[0][0] if c else None


chosen = [s for s in P["strategies"] if s in ("direct", "cot", "fewshot", "self_consistency")] or ["cot"]
results, examples, sc_curve = [], {}, []
for si, strategy in enumerate(chosen):
    n = int(P["k"]) if strategy == "self_consistency" else 1
    temp = float(P["temperature"]) if strategy == "self_consistency" else 0.0
    sys_prompt = DIRECT if strategy == "direct" else system
    prompts = [(f"{fewshot}\n\n{t['prompt']}" if strategy == "fewshot" else t["prompt"]) for t in tasks]
    groups = generate(model, tok, prompts, int(P["max_new_tokens"]), temp, num_samples=n, batch_size=max(16, n * 2), system=sys_prompt, progress=lambda d, t: progress(5 + 90 * (si + d / t) / len(chosen), f"{strategy}: {d}/{t}"))
    preds = [[world.extract(c) for c in g] for g in groups]
    correct = [world.grade("" if majority(p) is None else f"{world.answer_prefix} {majority(p)}", t["answer"]) for p, t in zip(preds, tasks)]
    tokens = sum(len(tok.encode(c)) for g in groups for c in g) / len(tasks)
    by_family = {}
    for c, t in zip(correct, tasks):
        f = by_family.setdefault(t["family"], [0, 0])
        f[0] += c
        f[1] += 1
    results.append({"strategy": strategy, "accuracy": sum(correct) / len(tasks), "tokens": tokens, "by_family": {k: v[0] / v[1] for k, v in by_family.items()}})
    examples[strategy] = [g[0][:300] for g in groups[:12]]
    if strategy == "self_consistency":
        for j in range(1, n + 1):
            acc = sum(world.grade(f"{world.answer_prefix} {majority(p[:j])}", t["answer"]) for p, t in zip(preds, tasks)) / len(tasks)
            sc_curve.append({"k": j, "accuracy": acc, "tokens": tokens * j / n})

best = max(results, key=lambda r: r["accuracy"])
families = sorted({f for r in results for f in r["by_family"]})
R = Result()
for r in results:
    R.metric(f"acc_{r['strategy']}", f"Accuracy · {r['strategy']}", r["accuracy"], "pct", "kept" if r is best else "sky", help=f"{r['tokens']:.0f} tokens per question")
R.metric("best_accuracy", "Best", best["accuracy"], "pct", "kept", help=best["strategy"])
R.chart("acc", "Accuracy by strategy", [{"strategy": r["strategy"], "accuracy": r["accuracy"]} for r in results], "strategy", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("cost", "Tokens generated per question", [{"strategy": r["strategy"], "tokens": r["tokens"]} for r in results], "strategy", [{"key": "tokens", "label": "Tokens", "color": "raw"}], "bar", note="Accuracy you buy with tokens is still accuracy, but the bill arrives on every question you ever ask.")
if sc_curve:
    R.chart("sc", "Self-consistency: accuracy against attempts", sc_curve, "k", [{"key": "accuracy", "label": "Majority accuracy", "color": "hold"}], "line", y_domain=[0, 1], note="The curve usually flattens by four or five attempts.")
R.chart("families", "Accuracy by family", [dict({"family": f}, **{r["strategy"]: r["by_family"].get(f, 0) for r in results}) for f in families], "family", [{"key": r["strategy"], "label": r["strategy"]} for r in results], "bar", y_domain=[0, 1])
R.table("examples", "What it wrote", [{"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}] + [{"key": s, "label": s} for s in chosen], [dict({"question": t["prompt"][:180], "gold": t["answer"]}, **{s: examples[s][i] for s in chosen}) for i, t in enumerate(tasks[:12])])
R.output("model", model_path).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("system", system).output("best_strategy", best["strategy"]).output("best_accuracy", best["accuracy"])
R.save()
