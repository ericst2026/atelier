"""Step 2 — accuracy and cost of prompting strategies."""
import json
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"model": "Qwen2.5-0.5B-Instruct", "strategies": ["direct", "cot", "fewshot", "self_consistency"], "k": 8, "temperature": 0.7, "max_new_tokens": 384})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
evals = hf.read_jsonl(I["eval"])
fewshot = json.loads((Path(__file__).resolve().parents[1] / "data" / "fewshot.json").read_text())
mp = hf.model_path(P["model"])
tok = hf.load_tokenizer(mp)
model = hf.load_model(mp)
COT = "Solve the problem step by step, then give the final answer on its own line as '#### <number>'."
DIRECT = "Answer with only the final number, nothing else."


def prompt_for(strategy: str, q: str) -> str:
    if strategy == "direct":
        return hf.chat_prompt(tok, q, DIRECT)
    if strategy == "fewshot":
        shots = "\n\n".join(f"Problem: {s['question']}\nSolution: {s['solution']}" for s in fewshot)
        return hf.chat_prompt(tok, f"{shots}\n\nProblem: {q}\nSolution:", COT)
    return hf.chat_prompt(tok, q, COT)


def majority(answers: list[str | None]) -> str | None:
    c = Counter(a for a in answers if a is not None)
    return c.most_common(1)[0][0] if c else None


results, examples = [], {}
sc_curve = []
strategies = [s for s in P["strategies"] if s in ("direct", "cot", "fewshot", "self_consistency")]
for si, strategy in enumerate(strategies):
    n = int(P["k"]) if strategy == "self_consistency" else 1
    temp = float(P["temperature"]) if strategy == "self_consistency" else 0.0
    prompts = [prompt_for(strategy, e["question"]) for e in evals]
    gens = hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), temp, batch_size=8 if n > 1 else 16, num_return_sequences=n, progress=lambda d, t: progress(5 + 90 * (si + d / t) / len(strategies), f"{strategy}: {d}/{t}"))
    tokens = sum(len(tok(g).input_ids) for gs in gens for g in gs) / len(evals)
    preds = [[hf.extract_answer(g) for g in gs] for gs in gens]
    correct = [hf.answers_equal(majority(p), e["gold"]) for p, e in zip(preds, evals)]
    by_diff = {}
    for c, e in zip(correct, evals):
        d = by_diff.setdefault(e["difficulty"], [0, 0])
        d[0] += c
        d[1] += 1
    results.append({"strategy": strategy, "accuracy": sum(correct) / len(evals), "tokens": tokens, "by_diff": {k: v[0] / v[1] for k, v in by_diff.items()}})
    examples[strategy] = [g[0][:400] for g in gens[:12]]
    if strategy == "self_consistency":
        for j in range(1, n + 1):
            sc_curve.append({"k": j, "accuracy": sum(hf.answers_equal(majority(p[:j]), e["gold"]) for p, e in zip(preds, evals)) / len(evals)})
    hf.write_jsonl(run_dir / f"predictions_{strategy}.jsonl", [{"question": e["question"], "gold": e["gold"], "preds": p, "correct": c, "completions": gs} for e, p, c, gs in zip(evals, preds, correct, gens)])

best = max(results, key=lambda r: r["accuracy"])
diffs = sorted({d for r in results for d in r["by_diff"]})
R = Result()
for r in results:
    R.metric(f"acc_{r['strategy']}", f"Accuracy · {r['strategy']}", r["accuracy"], "pct", "kept" if r is best else "sky", help=f"{r['tokens']:.0f} tokens/problem")
R.metric("best_acc", "Best accuracy", best["accuracy"], "pct", "kept", help=best["strategy"])
R.chart("acc", "Accuracy by strategy", [{"strategy": r["strategy"], "accuracy": r["accuracy"]} for r in results], "strategy", [{"key": "accuracy", "label": "Accuracy", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("tokens", "Generated tokens per problem", [{"strategy": r["strategy"], "tokens": r["tokens"]} for r in results], "strategy", [{"key": "tokens", "label": "Tokens", "color": "raw"}], "bar")
if sc_curve:
    R.chart("sc", "Self-consistency: accuracy vs samples", sc_curve, "k", [{"key": "accuracy", "label": "Majority accuracy", "color": "hold"}], "line", y_domain=[0, 1])
R.chart("diff", "Accuracy by difficulty", [dict({"difficulty": d}, **{r["strategy"]: r["by_diff"].get(d, 0) for r in results}) for d in diffs], "difficulty", [{"key": r["strategy"], "label": r["strategy"]} for r in results], "bar", y_domain=[0, 1])
R.table("examples", "Examples", [{"key": "question", "label": "Question"}, {"key": "gold", "label": "Gold"}] + [{"key": s, "label": s} for s in strategies], [dict({"question": e["question"][:250], "gold": e["gold"]}, **{s: examples[s][i] for s in strategies}) for i, e in enumerate(evals[:12])])
R.output("eval", I["eval"]).output("train", I["train"]).output("best_strategy", best["strategy"]).output("best_acc", best["accuracy"]).output("baseline_model", str(mp))
R.save()
