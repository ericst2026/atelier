"""Step 2 — find the band of prompts the policy sometimes gets right."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, write_jsonl
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_prompts": 2000, "families": ["arith", "count", "shop", "compare", "seq"], "difficulty": "mixed", "probe": 24, "group_size": 8})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world_all = World(lang=I.get("lang", "en"), seed=9)
families = list(P["families"]) or world_all.families
difficulty = None if P["difficulty"] == "mixed" else int(P["difficulty"])
model, _ = MiniLM.load(I["policy"], device)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or world_all.system_prompt

# probe every family and difficulty: pass rate 0 or 1 means no gradient signal
cells, rows = [], []
k = int(P["group_size"])
probe_n = int(P["probe"])
grid = [(f, d) for f in families for d in ([0, 1, 2, 3] if difficulty is None else [difficulty])]
for gi, (family, d) in enumerate(grid):
    tasks = world_all.eval_set(probe_n, seed=6_000 + gi, difficulty=d, families=[family])
    groups = generate(model, tok, [t["prompt"] for t in tasks], 160, 1.0, num_samples=k, batch_size=max(16, k * 2), system=system)
    pass_rates = [sum(world_all.grade(c, t["answer"]) for c in g) / k for t, g in zip(tasks, groups)]
    mean = sum(pass_rates) / len(pass_rates)
    learnable = sum(1 for r in pass_rates if 0 < r < 1) / len(pass_rates)
    cells.append({"family": family, "difficulty": d, "pass_rate": mean, "learnable": learnable})
    progress(5 + 80 * (gi + 1) / len(grid), f"{family} d{d}: pass {mean:.0%}, learnable {learnable:.0%}")

# weight the training mix towards the learnable band
weights = {(c["family"], c["difficulty"]): max(c["learnable"], 0.05) for c in cells}
total_w = sum(weights.values())
prompts = []
import random  # noqa: E402

rng = random.Random(77)
n = int(P["n_prompts"])
for (family, d), w in weights.items():
    share = max(1, int(n * w / total_w))
    for t in world_all.eval_set(share, seed=rng.randrange(1_000_000), difficulty=d, families=[family]):
        prompts.append({"prompt": t["prompt"], "answer": t["answer"], "family": t["family"], "difficulty": t["difficulty"], "steps": t["steps"]})
rng.shuffle(prompts)
prompts = prompts[:n]
write_jsonl(run_dir / "prompts.jsonl", prompts)

learnable_overall = sum(c["learnable"] for c in cells) / len(cells)
by_family = {}
for c in cells:
    by_family.setdefault(c["family"], []).append(c)

R = Result()
R.metric("prompts", "Training prompts", len(prompts), "int", "kept")
R.metric("learnable", "Prompts with a mixed group", learnable_overall, "pct", "sky", help="sometimes right, sometimes wrong — the only prompts GRPO can learn from")
R.metric("pass_rate", "Mean pass rate", sum(c["pass_rate"] for c in cells) / len(cells), "pct", "raw")
R.metric("families", "Families in the mix", len(families), "int", "hold")
R.chart("grid", "Pass rate by family and difficulty", [dict({"family": f}, **{f"d{c['difficulty']}": c["pass_rate"] for c in cs}) for f, cs in sorted(by_family.items())], "family", [{"key": f"d{d}", "label": f"difficulty {d}"} for d in ([0, 1, 2, 3] if difficulty is None else [difficulty])], "bar", y_domain=[0, 1])
R.chart("learnable", "Share of questions with a mixed group", [dict({"family": f}, **{f"d{c['difficulty']}": c["learnable"] for c in cs}) for f, cs in sorted(by_family.items())], "family", [{"key": f"d{d}", "label": f"difficulty {d}"} for d in ([0, 1, 2, 3] if difficulty is None else [difficulty])], "bar", y_domain=[0, 1], note="The training mix is weighted towards the tall bars. All-or-nothing cells contribute almost nothing to a GRPO update.")
R.table("cells", "Probe", [{"key": "family", "label": "Family"}, {"key": "difficulty", "label": "Difficulty"}, {"key": "pass_rate", "label": "Pass rate", "fmt": "pct"}, {"key": "learnable", "label": "Mixed groups", "fmt": "pct"}], cells)
R.artifact(run_dir / "prompts.jsonl", "prompts.jsonl")
R.output("prompts", str(run_dir / "prompts.jsonl")).output("families", families).output("learnable", learnable_overall)
for k_ in ("reward_spec", "policy", "tokenizer", "lang", "system", "sft_accuracy"):
    if k_ in I:
        R.output(k_, I[k_])
R.save()
