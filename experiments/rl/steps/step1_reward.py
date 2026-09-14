"""Step 1 — prepare prompt sets and the reward (verifiable, or a trained reward model)."""
import json
import os
import random
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"reward_type": "verifiable", "n_train": 2000, "n_test": 300, "rm_base": "Qwen2.5-0.5B-Instruct", "rm_pairs": 4000, "rm_epochs": 1.0, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
rng = random.Random(int(P["seed"]))

gsm_train = hf.dataset_split("rl", "gsm8k", "train")
gsm_test = hf.dataset_split("rl", "gsm8k", "test")
rng.shuffle(gsm_train)
rng.shuffle(gsm_test)
SYSTEM = "Solve the problem step by step, then give the final answer on its own line as '#### <number>'."
train_rows = [{"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}], "answer": hf.gsm8k_gold(r["answer"]), "question": r["question"]} for r in gsm_train[: int(P["n_train"])]]
test_rows = [{"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}], "answer": hf.gsm8k_gold(r["answer"]), "question": r["question"], "solution": r["answer"]} for r in gsm_test[: int(P["n_test"])]]
hf.write_jsonl(run_dir / "train_prompts.jsonl", train_rows)
hf.write_jsonl(run_dir / "test_prompts.jsonl", test_rows)
progress(15, f"{len(train_rows)} train / {len(test_rows)} test prompts")

R = Result()
spec = {"type": P["reward_type"], "system": SYSTEM}
if P["reward_type"] == "verifiable":
    mags = []
    for r in train_rows:
        try:
            mags.append(abs(float(r["answer"])) + 1)
        except ValueError:
            pass
    steps = [r["solution"].count("\n") + 1 for r in test_rows]
    R.metric("reward_type", "Reward", "verifiable", "text", "kept")
    R.metric("train_prompts", "Training prompts", len(train_rows), "int", "raw")
    R.metric("test_prompts", "Held-out prompts", len(test_rows), "int", "hold")
    R.chart("mags", "Answer magnitude (train)", hist(mags, bins=20, log=True), "bin", [{"key": "count", "label": "Problems", "color": "raw"}], "bar")
    R.chart("steps", "Reference solution length (lines, held-out)", hist(steps, bins=12, integer=True), "bin", [{"key": "count", "label": "Problems", "color": "hold"}], "bar")
    R.table("samples", "Samples", [{"key": "question", "label": "Question"}, {"key": "answer", "label": "Gold"}], [{"question": r["question"][:300], "answer": r["answer"]} for r in test_rows[:15]])
    R.note("Reward = 1.0 if the last '#### number' equals the gold answer, plus 0.2 for marking a final answer at all. Everything else is 0.")
else:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification
    from trl import RewardConfig, RewardTrainer

    pairs = hf.dataset_split("rl", "hh-rlhf", "train")
    rng.shuffle(pairs)
    pairs = [{"chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs[: int(P["rm_pairs"])] if p.get("chosen") and p.get("rejected")]
    n_val = max(50, len(pairs) // 20)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    mp = hf.model_path(P["rm_base"])
    tok = hf.load_tokenizer(mp, padding_side="right")
    model = AutoModelForSequenceClassification.from_pretrained(str(mp), num_labels=1, torch_dtype=torch.bfloat16, local_files_only=True)
    model.config.pad_token_id = tok.pad_token_id
    steps_per_epoch = max(1, len(train_pairs) // 16)
    total = int(steps_per_epoch * float(P["rm_epochs"]))
    cb = hf.ProgressCallback(total, "reward model")
    cfg = RewardConfig(output_dir=str(run_dir / "trainer"), num_train_epochs=float(P["rm_epochs"]), per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=1e-5, max_length=1024, logging_steps=5, eval_strategy="no", save_strategy="no", bf16=True, report_to="none")
    trainer = RewardTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(train_pairs), processing_class=tok, callbacks=[cb.callback])
    trainer.train()
    rm_dir = run_dir / "reward_model"
    trainer.save_model(str(rm_dir))
    tok.save_pretrained(str(rm_dir))
    progress(85, "scoring held-out pairs")
    model.eval()
    dev = next(model.parameters()).device

    def score(texts):
        out = []
        for i in range(0, len(texts), 8):
            enc = tok(texts[i : i + 8], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(dev)
            with torch.no_grad():
                out += model(**enc).logits.squeeze(-1).float().tolist()
        return out

    sc = score([p["chosen"] for p in val_pairs])
    sr = score([p["rejected"] for p in val_pairs])
    acc = sum(1 for a, b in zip(sc, sr) if a > b) / len(val_pairs)
    spec.update({"reward_model": str(rm_dir), "accuracy": acc})
    R.metric("reward_type", "Reward", "reward model", "text", "kept")
    R.metric("rm_accuracy", "Pairwise accuracy (held-out)", acc, "pct", "kept", help=f"{len(val_pairs)} pairs")
    R.metric("rm_pairs", "Training pairs", len(train_pairs), "int", "raw")
    R.chart("scores", "Reward-model scores", [dict(b, chosen=b["count"]) for b in hist(sc, bins=20)], "bin", [{"key": "chosen", "label": "Chosen", "color": "kept"}], "bar", note="Scores of the held-out chosen responses; rejected responses should sit lower on average.")
    R.chart("margin", "Score margin (chosen − rejected)", hist([a - b for a, b in zip(sc, sr)], bins=20), "bin", [{"key": "count", "label": "Pairs", "color": "sky"}], "bar", ref_x=None)
    R.chart("loss", "Reward-model training loss", hf.history_chart(cb.history, ["loss"]), "step", [{"key": "loss", "label": "Loss", "color": "kept"}], "line")
    R.artifact(rm_dir, "reward model")
(run_dir / "reward_spec.json").write_text(json.dumps(spec, indent=2))
R.artifact(run_dir / "train_prompts.jsonl", "train_prompts.jsonl").artifact(run_dir / "test_prompts.jsonl", "test_prompts.jsonl").artifact(run_dir / "reward_spec.json", "reward_spec.json")
R.output("reward_spec", str(run_dir / "reward_spec.json")).output("train_prompts", str(run_dir / "train_prompts.jsonl")).output("test_prompts", str(run_dir / "test_prompts.jsonl")).output("reward_type", P["reward_type"])
R.save()
