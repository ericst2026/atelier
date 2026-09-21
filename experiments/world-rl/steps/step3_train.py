"""Step 3 — GRPO."""
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, curves, read_jsonl
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
fmt = I.get("model_format", "atelier")
max_steps = int(P["max_steps"])


def reward_fn(task, completion):
    return shaped_reward(world, task, completion, weights)["total"]


if fmt == "hf":
    # A HuggingFace policy: TRL's GRPO, with the same shaped reward. It trains the whole
    # model (the reference is a frozen copy), so a LoRA adapter from fine-tuning is merged
    # into a starting copy first and the result is a full model, not an adapter.
    from atelier_nlp import hf

    system = I.get("system") or spec.get("system") or world.system_prompt
    start = I["policy"]
    if I.get("adapter"):
        progress(1, "merging the fine-tuning adapter into a starting copy")
        merged = hf.load_model(I["policy"], dtype="bf16" if device == "cuda" else "fp32", device="cpu", adapter=I["adapter"])
        start = str(run_dir / "start")
        merged.save_pretrained(start)
        hf.load_tokenizer(I["policy"]).save_pretrained(start)
        del merged
    progress(2, f"{len(prompts)} prompts · group {P['group_size']} · {max_steps} steps · GRPO on {I.get('model_label', I['policy'])} · {device}")

    def grpo_reward(prompts=None, completions=None, answer=None, **_):
        return [reward_fn({"answer": a}, hf.completion_text(c)) for c, a in zip(completions, answer)]

    rows = [{"prompt": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p["prompt"]}], "answer": str(p["answer"])} for p in prompts]
    res = hf.grpo_train(Path(start), rows, [grpo_reward], run_dir, num_generations=int(P["group_size"]), max_completion_length=int(P["max_new_tokens"]), temperature=float(P["temperature"]), beta=float(P["beta"]), lr=float(P["lr"]), max_steps=max_steps, prompts_per_step=int(P["prompts_per_step"]), dtype="bf16" if device == "cuda" else "fp32", label="GRPO")
    if start != I["policy"]:
        shutil.rmtree(start, ignore_errors=True)
    h = [{"step": x["step"], "reward": x["reward"], "kl": x.get("kl", 0.0), "completion_length": x.get("completions/mean_length", x.get("completion_length", 0.0)), "reward_std": x.get("reward_std", 0.0)} for x in res["history"] if "reward" in x]
    rl_model = res["save_dir"]
else:
    from atelier_mini.model import MiniLM
    from atelier_mini.rl import grpo
    from atelier_mini.tok import MiniTokenizer

    model, _ = MiniLM.load(I["policy"], device)
    tok = MiniTokenizer.load(I["tokenizer"])
    system = I.get("system") or spec.get("system") or world.system_prompt

    reference = None
    if float(P["beta"]) > 0:
        reference, _ = MiniLM.load(I["policy"], device)
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)

    progress(2, f"{len(prompts)} prompts · group {P['group_size']} · {max_steps} steps · {device}")

    def on_log(row):
        progress(100 * (row["step"] + 1) / max_steps, f"step {row['step'] + 1}/{max_steps} · reward {row['reward']:.3f} · KL {row['kl']:.4f} · {row['completion_length']:.0f} tokens",
                 step=row["step"], **curves(row, "reward", "reward_std", "kl", "completion_length", "loss", "grad_norm", "solved"))

    res = grpo(model, tok, prompts, reward_fn, run_dir, reference=reference, group_size=int(P["group_size"]), prompts_per_step=int(P["prompts_per_step"]), max_new_tokens=int(P["max_new_tokens"]), temperature=float(P["temperature"]), beta=float(P["beta"]), lr=float(P["lr"]), max_steps=max_steps, clip_eps=float(P["clip_eps"]), system=system, device=device, on_log=on_log)
    model.save(run_dir / "model.pt", {"rl": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["policy"]})
    h = res["history"]
    rl_model = str(run_dir / "model.pt")
if not h:
    raise SystemExit("Training logged no reward. Raise the number of optimizer steps.")
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
if fmt == "hf":
    R.note(f"GRPO with TRL on {I.get('model_label', 'the HuggingFace policy')}; the policy after RL is a full model in {Path(rl_model).name}/. The clip range is TRL's default here.")
else:
    R.artifact(run_dir / "model.pt", "model.pt (after RL)")
R.output("rl_model", rl_model).output("final_reward", last)
for k in ("policy", "tokenizer", "reward_spec", "lang", "system", "families", "sft_accuracy", "model_format", "adapter", "model_label", "data_source", "qa_val"):
    if k in I:
        R.output(k, I[k])
R.save()
