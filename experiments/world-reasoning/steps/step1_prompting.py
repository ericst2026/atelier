"""Step 1 — what each way of asking is worth, and what it costs."""
import json
import os
from collections import Counter
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, write_jsonl
from atelier_world import World
from atelier_world.prepared import choose_model, load_lm, model_outputs, read_qa, train_val

parse_args()
P = params({"model_source": "generated", "lang": "en", "data_source": "generated", "n_eval": 250, "strategies": ["direct", "cot", "fewshot", "self_consistency"], "k": 8, "temperature": 0.8, "max_new_tokens": 192})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
# a Fine-tuning run names its model sft_model, a Pretraining run names it model
run_outputs = (I.get("model_run_run") or {}).get("outputs") or {}
info = choose_model(P, I, run_key="model_run", run_model_key="sft_model" if run_outputs.get("sft_model") else "model", hint="Choose a model: a Fine-tuning step 3 run, a Pretraining step 3 run, or a prepared model.")
model_path = info["model"]
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=info["lang"], seed=3)
lm = load_lm(info, device)
tok = lm.tok
system = lm.system or world.system_prompt
prepared = P["data_source"] == "prepared"
if prepared:
    # training rows feed step 2; the validation split (or a held-out slice) is asked here and in step 4
    progress(3, f"reading materials/{P['data_material']}")
    train_qa, val_qa = train_val(P["data_material"], read_qa, 1000, seed=3, limit=40000)
    write_jsonl(run_dir / "qa_train.jsonl", train_qa)
    write_jsonl(run_dir / "qa_val.jsonl", val_qa)
    tasks = val_qa[: int(P["n_eval"])]
    shots = train_qa[:2]
    data_label = f"materials/{P['data_material']}"
else:
    tasks = world.eval_set(int(P["n_eval"]), seed=222_004)
    shots = world.eval_set(2, seed=13)
    data_label = "generated from the world"
fewshot = "\n\n".join(f"{s['prompt']}\n" + "".join(f"{line}\n" for line in s["steps"]) + f"{world.answer_prefix} {s['answer']}" for s in shots)
DIRECT = world.pack["instruction_system"] + " " + ("Give only the final answer." if world.lang == "en" else "答えだけを書いてください。")


def majority(answers):
    c = Counter(a for a in answers if a)
    return c.most_common(1)[0][0] if c else None


def n_tokens(text):
    return len(tok.encode(text, add_special_tokens=False)) if info["format"] == "hf" else len(tok.encode(text))


chosen = [s for s in P["strategies"] if s in ("direct", "cot", "fewshot", "self_consistency")] or ["cot"]
results, examples, sc_curve = [], {}, []
for si, strategy in enumerate(chosen):
    n = int(P["k"]) if strategy == "self_consistency" else 1
    temp = float(P["temperature"]) if strategy == "self_consistency" else 0.0
    sys_prompt = DIRECT if strategy == "direct" else system
    prompts = [(f"{fewshot}\n\n{t['prompt']}" if strategy == "fewshot" else t["prompt"]) for t in tasks]
    groups = lm.generate(prompts, int(P["max_new_tokens"]), temp, num_samples=n, batch_size=max(16, n * 2), system=sys_prompt, progress=lambda d, t: progress(5 + 90 * (si + d / t) / len(chosen), f"{strategy}: {d}/{t}"))
    preds = [[world.extract(c) for c in g] for g in groups]
    correct = [world.grade("" if majority(p) is None else f"{world.answer_prefix} {majority(p)}", t["answer"]) for p, t in zip(preds, tasks)]
    tokens = sum(n_tokens(c) for g in groups for c in g) / len(tasks)
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
(run_dir / "reasoning_meta.json").write_text(json.dumps({"data_source": P["data_source"], "data_label": data_label, "model_label": info["label"]}, ensure_ascii=False, indent=2))
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
for key, v in model_outputs(info, "model").items():
    R.output(key, v)
R.output("system", system).output("best_strategy", best["strategy"]).output("best_accuracy", best["accuracy"]).output("data_source", P["data_source"])
if prepared:
    R.output("qa_train", str(run_dir / "qa_train.jsonl")).output("qa_val", str(run_dir / "qa_val.jsonl"))
    R.note(f"Questions from {data_label}: the held-out ones here ({len(val_qa):,} available), {len(train_qa):,} training rows for step 2; the two solved examples are training rows. Model: {info['label']} ({info['format']}).")
else:
    R.note(f"Questions generated from the world. Model: {info['label']} ({info['format']}).")
R.save()
