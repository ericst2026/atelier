"""Step 4 — pass@k before/after, by difficulty, and reasoning length."""
import math
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"n_samples": 8, "temperature": 0.7, "max_new_tokens": 384})
I = inputs()
evals = hf.read_jsonl(I["eval"])
n = int(P["n_samples"])
COT = "Solve the problem step by step, then give the final answer on its own line as '#### <number>'."


def pass_at_k(n_, c, k):
    if n_ - c < k:
        return 1.0
    return 1.0 - math.comb(n_ - c, k) / math.comb(n_, k)


out = {}
for j, (name, adapter) in enumerate((("before", None), ("after", I["adapter"]))):
    tok = hf.load_tokenizer(I["base_model"])
    model = hf.load_model(I["base_model"], adapter=adapter)
    prompts = [hf.chat_prompt(tok, e["question"], COT) for e in evals]
    gens = hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), float(P["temperature"]), batch_size=8, num_return_sequences=n, progress=lambda d, t: progress(5 + 45 * j + 40 * d / t, f"{name}: {d}/{t}"))
    corr = [[hf.answers_equal(hf.extract_answer(g), e["gold"]) for g in gs] for gs, e in zip(gens, evals)]
    lens_ok = [len(g) for gs, cs in zip(gens, corr) for g, c in zip(gs, cs) if c]
    lens_bad = [len(g) for gs, cs in zip(gens, corr) for g, c in zip(gs, cs) if not c]
    by_diff = {}
    for cs, e in zip(corr, evals):
        d = by_diff.setdefault(e["difficulty"], [0.0, 0])
        d[0] += sum(cs) / n
        d[1] += 1
    out[name] = {"pass": [sum(pass_at_k(n, sum(cs), k) for cs in corr) / len(evals) for k in range(1, n + 1)], "by_diff": {d: v[0] / v[1] for d, v in by_diff.items()}, "lens_ok": lens_ok, "lens_bad": lens_bad, "first": [gs[0][:400] for gs in gens[:10]], "corr": corr}
    del model
    import torch

    torch.cuda.empty_cache()

b, a = out["before"], out["after"]
diffs = sorted(set(b["by_diff"]) | set(a["by_diff"]))
R = Result()
R.metric("acc_before", "pass@1 before", b["pass"][0], "pct", "raw")
R.metric("acc_after", "pass@1 after", a["pass"][0], "pct", "kept")
R.metric("passk_after", f"pass@{n} after", a["pass"][-1], "pct", "hold", help=f"before: {100 * b['pass'][-1]:.1f}%")
R.metric("len_ok", "Correct chain length (chars)", sum(a["lens_ok"]) / max(1, len(a["lens_ok"])), "num", "sky", help=f"wrong chains: {sum(a['lens_bad']) / max(1, len(a['lens_bad'])):.0f}")
R.chart("passk", "pass@k", [{"k": k + 1, "before": b["pass"][k], "after": a["pass"][k]} for k in range(n)], "k", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "line", y_domain=[0, 1])
R.chart("diff", "Mean accuracy by difficulty", [{"difficulty": d, "before": b["by_diff"].get(d, 0), "after": a["by_diff"].get(d, 0)} for d in diffs], "difficulty", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1])
hi = max(a["lens_ok"] + a["lens_bad"] + [1])
lo = {x["bin"]: x["count"] for x in hist(a["lens_ok"], bins=15, lo=0, hi=hi)}
R.chart("lens", "Chain length: correct vs wrong (after)", [{"bin": x["bin"], "correct": lo.get(x["bin"], 0), "wrong": x["count"]} for x in hist(a["lens_bad"], bins=15, lo=0, hi=hi)], "bin", [{"key": "correct", "label": "Correct", "color": "kept"}, {"key": "wrong", "label": "Wrong", "color": "dup"}], "bar")
R.table("examples", "First sample per problem", [{"key": "question", "label": "Question"}, {"key": "gold", "label": "Gold"}, {"key": "before", "label": "Before"}, {"key": "after", "label": "After"}], [{"question": e["question"][:250], "gold": e["gold"], "before": ("✓ " if b["corr"][i][0] else "✗ ") + b["first"][i], "after": ("✓ " if a["corr"][i][0] else "✗ ") + a["first"][i]} for i, e in enumerate(evals[:10])])
R.output("accuracy", a["pass"][0]).output("accuracy_before", b["pass"][0])
R.save()
