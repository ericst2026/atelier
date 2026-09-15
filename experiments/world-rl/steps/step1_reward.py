"""Step 1 — define the reward and look at what it pays for before training on it."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_world import World
from atelier_world.prepared import choose_model, load_lm, model_outputs, read_qa, train_val

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_reward import shaped_reward  # noqa: E402

parse_args()
P = params({"model_source": "generated", "lang": "en", "data_source": "generated", "w_correct": 1.0, "w_marked": 0.2, "w_steps": 0.1, "w_length": 0.05, "w_repeat": 0.5, "n_probe": 80, "group_size": 8, "temperature": 1.0})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
info = choose_model(P, I, run_key="sft_run", run_model_key="sft_model", hint="Choose a Fine-tuning run (step 3) as the starting policy, or a prepared model.")
o = info["outputs"]
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = {"correct": float(P["w_correct"]), "marked": float(P["w_marked"]), "steps": float(P["w_steps"]), "length": float(P["w_length"]), "repeat": float(P["w_repeat"])}
world = World(lang=info["lang"], seed=5)
lm = load_lm(info, device)
system = lm.system or world.system_prompt
prepared = P["data_source"] == "prepared"

if prepared:
    # the questions come from the teacher's dataset: training rows for steps 1–3,
    # its validation split (or a held-out slice) for step 4
    progress(5, f"reading materials/{P['data_material']}")
    train_qa, val_qa = train_val(P["data_material"], read_qa, 2000, seed=5, limit=20000)
    write_jsonl(run_dir / "qa_train.jsonl", train_qa)
    write_jsonl(run_dir / "qa_val.jsonl", val_qa)
    tasks = train_qa[: int(P["n_probe"])]
    data_label = f"materials/{P['data_material']}"
else:
    tasks = world.eval_set(int(P["n_probe"]), seed=31_337)
    data_label = "generated from the world"
k = int(P["group_size"])
progress(10, f"sampling {k} answers for each of {len(tasks)} questions")
groups = lm.generate([t["prompt"] for t in tasks], 160, float(P["temperature"]), num_samples=k, batch_size=max(16, k * 2), system=system, progress=lambda d, t: progress(10 + 70 * d / t, f"{d}/{t}"))

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
spec = {"weights": weights, "system": system, "data_source": P["data_source"], "data_label": data_label, "model_label": info["label"]}
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
R.output("reward_spec", str(run_dir / "reward_spec.json"))
for key, v in model_outputs(info, "policy").items():
    R.output(key, v)
R.output("system", system).output("sft_accuracy", o.get("accuracy")).output("data_source", P["data_source"])
if prepared:
    R.output("qa_train", str(run_dir / "qa_train.jsonl")).output("qa_val", str(run_dir / "qa_val.jsonl"))
    R.note(f"Questions from {data_label}: {len(train_qa):,} for probing and training, {len(val_qa):,} held out for step 4. Starting policy: {info['label']} ({info['format']}).")
else:
    R.note(f"Questions generated from the world. Starting policy: {info['label']} ({info['format']}).")
R.save()
