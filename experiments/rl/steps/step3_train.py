"""Step 3 — GRPO training with the reward from step 1."""
import json
import os
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"dtype": "bf16"})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
cfg = json.loads(Path(I["policy_config"]).read_text())
spec = json.loads(Path(I["reward_spec"]).read_text())
rows = hf.read_jsonl(I["train_prompts"])
init = cfg["init"]
policy_path = Path(init["model_dir"])
if init.get("adapter"):
    progress(2, "merging the SFT adapter into the base model")
    merged = hf.load_model(init["base_model"], adapter=init["adapter"])
    policy_path = run_dir / "init_policy"
    merged.save_pretrained(str(policy_path))
    hf.load_tokenizer(init["base_model"]).save_pretrained(str(policy_path))
    del merged

if spec["type"] == "verifiable":

    def correctness(completions, answer, **kw):
        return [1.0 if hf.answers_equal(hf.extract_answer(hf.completion_text(c)), a) else 0.0 for c, a in zip(completions, answer)]

    def formatting(completions, **kw):
        return [hf.format_reward(hf.completion_text(c)) for c in completions]

    reward_funcs = [correctness, formatting]
else:
    import torch
    from transformers import AutoModelForSequenceClassification

    rm_tok = hf.load_tokenizer(spec["reward_model"], padding_side="right")
    rm = AutoModelForSequenceClassification.from_pretrained(spec["reward_model"], torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()

    def reward_model(prompts, completions, **kw):
        texts = ["\n\nHuman: " + prompts[i][-1]["content"] + "\n\nAssistant: " + hf.completion_text(c) for i, c in enumerate(completions)]
        out = []
        for i in range(0, len(texts), 8):
            enc = rm_tok(texts[i : i + 8], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(rm.device)
            with torch.no_grad():
                out += rm(**enc).logits.squeeze(-1).float().tolist()
        return out

    reward_funcs = [reward_model]

progress(5, f"GRPO on {len(rows)} prompts from {init['label']}")
res = hf.grpo_train(policy_path, rows, reward_funcs, run_dir, num_generations=int(cfg["num_generations"]), max_completion_length=int(cfg["max_completion_length"]), temperature=float(cfg["temperature"]), beta=float(cfg["beta"]), lr=float(cfg["lr"]), max_steps=int(cfg["max_steps"]), prompts_per_step=int(cfg["prompts_per_step"]), dtype=P["dtype"])
h = res["history"]


def last(key):
    return next((x[key] for x in reversed(h) if key in x), None)


def series(*keys):
    rows_ = []
    for x in h:
        r = {"step": x.get("step", 0)}
        for k in keys:
            if k in x:
                r[k.replace("/", "_")] = x[k]
        if len(r) > 1:
            rows_.append(r)
    return rows_


len_key = next((k for k in ("completions/mean_length", "completion_length") if any(k in x for x in h)), "completion_length")
R = Result()
R.metric("final_reward", "Mean reward (last step)", last("reward"), "num", "kept")
R.metric("final_kl", "KL to reference (last step)", last("kl"), "num", "hold")
R.metric("final_length", "Completion length (last step)", last(len_key), "num", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("reward", "Reward", series("reward", "reward_std"), "step", [{"key": "reward", "label": "Mean reward", "color": "kept"}, {"key": "reward_std", "label": "Std within group", "color": "raw"}], "line")
R.chart("kl", "KL to the reference policy", series("kl"), "step", [{"key": "kl", "label": "KL", "color": "hold"}], "line")
R.chart("length", "Completion length", series(len_key), "step", [{"key": len_key.replace("/", "_"), "label": "Tokens", "color": "sky"}], "line")
R.artifact(Path(res["save_dir"]), "trained policy")
R.output("policy_dir", res["save_dir"]).output("init_policy", str(policy_path)).output("reward_spec", I["reward_spec"]).output("test_prompts", I["test_prompts"]).output("final_reward", last("reward"))
R.save()
