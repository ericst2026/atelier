"""Step 3 — GRPO."""
import copy
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.model import MiniLM
from atelier_mini.rl import grpo
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_reward import shaped_reward  # noqa: E402

parse_args()
P = params({"max_steps": 300, "group_size": 8, "prompts_per_step": 8, "max_new_tokens": 160, "temperature": 1.0, "beta": 0.02, "lr": 1e-5, "clip_eps": 0.2})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
spec = json.loads(Path(I["reward_spec"]).read_text())
weights = spec["weights"]
world = World(lang=I.get("lang", "en"), seed=5)
prompts = read_jsonl(I["prompts"])
model, _ = MiniLM.load(I["policy"], device)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or spec.get("system") or world.system_prompt

reference = None
if float(P["beta"]) > 0:
    reference, _ = MiniLM.load(I["policy"], device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

max_steps = int(P["max_steps"])
progress(2, f"{len(prompts)} prompts · group {P['group_size']} · {max_steps} steps · {device}")


def reward_fn(task, completion):
    return shaped_reward(world, task, completion, weights)["total"]


def on_log(row):
    progress(100 * (row["step"] + 1) / max_steps, f"step {row['step'] + 1}/{max_steps} · reward {row['reward']:.3f} · KL {row['kl']:.4f} · {row['completion_length']:.0f} tokens",
             step=row["step"], reward=row["reward"], kl=row["kl"], completion_length=row["completion_length"], reward_std=row["reward_std"])


res = grpo(model, tok, prompts, reward_fn, run_dir, reference=reference, group_size=int(P["group_size"]), prompts_per_step=int(P["prompts_per_step"]), max_new_tokens=int(P["max_new_tokens"]), temperature=float(P["temperature"]), beta=float(P["beta"]), lr=float(P["lr"]), max_steps=max_steps, clip_eps=float(P["clip_eps"]), system=system, device=device, on_log=on_log)
model.save(run_dir / "model.pt", {"rl": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["policy"]})
h = res["history"]
window = max(1, len(h) // 10)
first = sum(x["reward"] for x in h[:window]) / window
last = sum(x["reward"] for x in h[-window:]) / window

R = Result()
R.metric("final_reward", "Mean reward (last tenth)", last, "num", "kept", help=f"started at {first:.3f}")
R.metric("final_kl", "KL to the starting policy", sum(x["kl"] for x in h[-window:]) / window, "num", "hold")
R.metric("final_length", "Answer length (tokens)", sum(x["completion_length"] for x in h[-window:]) / window, "num", "sky", help=f"started at {sum(x['completion_length'] for x in h[:window]) / window:.0f}")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("reward", "Reward", [{"step": x["step"], "reward": x["reward"], "spread": x["reward_std"]} for x in h], "step", [{"key": "reward", "label": "Mean reward", "color": "kept"}, {"key": "spread", "label": "Spread within groups", "color": "raw"}], "line", note="When the spread collapses, the groups have stopped disagreeing and there is nothing left to learn from these prompts.")
R.chart("kl", "Distance from the starting policy", [{"step": x["step"], "kl": x["kl"]} for x in h], "step", [{"key": "kl", "label": "KL", "color": "hold"}], "line")
R.chart("length", "Answer length", [{"step": x["step"], "tokens": x["completion_length"]} for x in h], "step", [{"key": "tokens", "label": "Tokens", "color": "sky"}], "line", note="Rising length with flat reward is the classic sign that the shaping, not the answer, is being optimised.")
R.artifact(run_dir / "model.pt", "model.pt (after RL)")
R.output("rl_model", str(run_dir / "model.pt")).output("final_reward", last)
for k in ("policy", "tokenizer", "reward_spec", "lang", "system", "families", "sft_accuracy"):
    if k in I:
        R.output(k, I[k])
R.save()
