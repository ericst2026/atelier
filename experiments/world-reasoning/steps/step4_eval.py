"""Step 4 — pass@k before and after."""
import math
import os
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_eval": 250, "n_samples": 8, "temperature": 0.8, "max_new_tokens": 192})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=3)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or world.system_prompt
tasks = world.eval_set(int(P["n_eval"]), seed=314_159_265)
n = int(P["n_samples"])


def pass_at_k(total, correct, k):
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


out = {}
for j, (name, path) in enumerate((("before", I["model"]), ("after", I["rft_model"]))):
    model, _ = MiniLM.load(path, device)
    groups = generate(model, tok, [t["prompt"] for t in tasks], int(P["max_new_tokens"]), float(P["temperature"]), num_samples=n, batch_size=max(16, n * 2), system=system, progress=lambda d, t: progress(5 + 45 * j + 40 * d / t, f"{name}: {d}/{t}"))
    corr = [[world.grade(c, t["answer"]) for c in g] for g, t in zip(groups, tasks)]
    by_family = {}
    for cs, t in zip(corr, tasks):
        f = by_family.setdefault(t["family"], [0.0, 0])
        f[0] += sum(cs) / n
        f[1] += 1
    lens_ok = [len(c) for g, cs in zip(groups, corr) for c, ok in zip(g, cs) if ok]
    lens_bad = [len(c) for g, cs in zip(groups, corr) for c, ok in zip(g, cs) if not ok]
    out[name] = {
        "pass": [sum(pass_at_k(n, sum(cs), k) for cs in corr) / len(tasks) for k in range(1, n + 1)],
        "by_family": {k: v[0] / v[1] for k, v in by_family.items()},
        "lens_ok": lens_ok, "lens_bad": lens_bad,
        "first": [g[0] for g in groups], "corr": corr,
    }
    del model
    torch.cuda.empty_cache()

b, a = out["before"], out["after"]
families = sorted(set(b["by_family"]) | set(a["by_family"]))
R = Result()
R.metric("pass1_after", "pass@1 after", a["pass"][0], "pct", "kept")
R.metric("pass1_before", "pass@1 before", b["pass"][0], "pct", "raw")
R.metric("passk_after", f"pass@{n} after", a["pass"][-1], "pct", "hold", help=f"before: {b['pass'][-1]:.1%}")
R.metric("reliability", "pass@1 as a share of pass@k", a["pass"][0] / max(a["pass"][-1], 1e-9), "pct", "sky", help="how often the model lands an answer it can reach at all")
R.metric("chain_len", "Length of a correct chain", sum(a["lens_ok"]) / max(1, len(a["lens_ok"])), "num", "hold", help=f"wrong chains: {sum(a['lens_bad']) / max(1, len(a['lens_bad'])):.0f} characters")
R.chart("passk", "pass@k", [{"k": k + 1, "before": b["pass"][k], "after": a["pass"][k]} for k in range(n)], "k", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "line", y_domain=[0, 1], note="Training on its own correct attempts lifts pass@1 towards the old pass@k: the same reach, landed more reliably. If pass@k falls, the model has narrowed.")
R.chart("families", "Mean accuracy by family", [{"family": f, "before": b["by_family"].get(f, 0), "after": a["by_family"].get(f, 0)} for f in families], "family", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1])
hi = max(a["lens_ok"] + a["lens_bad"] + [1])
ok_bins = {x["bin"]: x["count"] for x in hist(a["lens_ok"], bins=15, lo=0, hi=hi)}
R.chart("lengths", "Chain length, correct against wrong", [{"bin": x["bin"], "correct": ok_bins.get(x["bin"], 0), "wrong": x["count"]} for x in hist(a["lens_bad"], bins=15, lo=0, hi=hi)], "bin", [{"key": "correct", "label": "Correct", "color": "kept"}, {"key": "wrong", "label": "Wrong", "color": "dup"}], "bar")
R.table("examples", "First attempt at each question", [{"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "before", "label": "Before"}, {"key": "after", "label": "After"}], [{"family": t["family"], "question": t["prompt"][:170], "gold": t["answer"], "before": ("✓ " if b["corr"][i][0] else "✗ ") + b["first"][i][:200], "after": ("✓ " if a["corr"][i][0] else "✗ ") + a["first"][i][:200]} for i, t in enumerate(tasks[:15])])
R.output("accuracy", a["pass"][0]).output("rft_model", I["rft_model"]).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en"))
R.save()
