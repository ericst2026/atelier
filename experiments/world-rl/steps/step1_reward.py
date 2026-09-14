"""Step 1 — define the reward and look at what it pays for before training on it."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_reward import shaped_reward  # noqa: E402

parse_args()
P = params({"w_correct": 1.0, "w_marked": 0.2, "w_steps": 0.1, "w_length": 0.05, "w_repeat": 0.5, "n_probe": 80, "group_size": 8, "temperature": 1.0})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("sft_run_run")
if not ref:
    raise SystemExit("Choose a Fine-tuning run (step 3) as the starting policy.")
o = ref["outputs"]
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = {"correct": float(P["w_correct"]), "marked": float(P["w_marked"]), "steps": float(P["w_steps"]), "length": float(P["w_length"]), "repeat": float(P["w_repeat"])}
world = World(lang=o.get("lang", "en"), seed=5)
model, ck = MiniLM.load(o["sft_model"], device)
tok = MiniTokenizer.load(o["tokenizer"])
system = o.get("system") or ck.get("system") or world.system_prompt

tasks = world.eval_set(int(P["n_probe"]), seed=31_337)
k = int(P["group_size"])
progress(10, f"sampling {k} answers for each of {len(tasks)} questions")
groups = generate(model, tok, [t["prompt"] for t in tasks], 160, float(P["temperature"]), num_samples=k, batch_size=max(16, k * 2), system=system, progress=lambda d, t: progress(10 + 70 * d / t, f"{d}/{t}"))

rows, spreads, totals, parts_sum = [], [], [], {"correct": 0.0, "marked": 0.0, "working": 0.0, "length_pen": 0.0, "repeat_pen": 0.0}
flat_correct = 0
for t, group in zip(tasks, groups):
    scores = []
    for c in group:
        parts = shaped_reward(world, t, c, weights)
        scores.append(parts["total"])
        totals.append(parts["total"])
        for key in parts_sum:
            parts_sum[key] += parts[key]
    spread = max(scores) - min(scores)
    spreads.append(spread)
    if spread < 1e-6:
        flat_correct += 1
    if len(rows) < 20:
        best = group[scores.index(max(scores))]
        worst = group[scores.index(min(scores))]
        rows.append({"family": t["family"], "question": t["prompt"][:180], "gold": t["answer"], "best": f"[{max(scores):.2f}] {best[:200]}", "worst": f"[{min(scores):.2f}] {worst[:200]}"})

n = len(totals)
flat_share = flat_correct / len(tasks)
spec = {"weights": weights, "system": system}
(run_dir / "reward_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))

R = Result()
R.metric("mean_reward", "Mean reward", sum(totals) / n, "num", "kept")
R.metric("spread", "Reward spread within a group", sum(spreads) / len(spreads), "num", "sky", help="GRPO learns from differences inside a group; a spread near zero is a wasted step")
R.metric("flat_groups", "Groups where every answer scored the same", flat_share, "pct", "dup" if flat_share > 0.5 else "hold", help="too easy or too hard — fix it on the next page")
R.metric("accuracy", "Answers correct while sampling", parts_sum["correct"] / n, "pct", "raw", help=f"at temperature {P['temperature']}")
R.chart("hist", "Reward distribution", hist(totals, bins=20), "bin", [{"key": "count", "label": "Answers", "color": "kept"}], "bar")
R.chart("spread", "Spread within each group", hist(spreads, bins=20), "bin", [{"key": "count", "label": "Questions", "color": "sky"}], "bar", note="Questions on the left teach nothing: every answer in the group scored alike.")
R.chart("parts", "Where the reward came from", [
    {"part": "correct", "value": weights["correct"] * parts_sum["correct"] / n},
    {"part": "marked", "value": weights["marked"] * parts_sum["marked"] / n},
    {"part": "working", "value": weights["steps"] * parts_sum["working"] / n},
    {"part": "length penalty", "value": -weights["length"] * parts_sum["length_pen"] / n},
    {"part": "repetition penalty", "value": -weights["repeat"] * parts_sum["repeat_pen"] / n},
], "part", [{"key": "value", "label": "Mean contribution", "color": "raw"}], "bar", note="If correctness is not the tallest bar, the policy will optimise the others first.")
R.table("pairs", "Best and worst answer in a group", [{"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "best", "label": "Highest scoring"}, {"key": "worst", "label": "Lowest scoring"}], rows)
R.artifact(run_dir / "reward_spec.json", "reward_spec.json")
R.output("reward_spec", str(run_dir / "reward_spec.json")).output("policy", o["sft_model"]).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("system", system).output("sft_accuracy", o.get("accuracy"))
R.save()
