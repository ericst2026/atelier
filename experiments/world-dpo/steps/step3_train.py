"""Step 3 — DPO."""
import math
import os
import time
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, curves, read_jsonl

parse_args()
P = params({"beta": 0.1, "epochs": 1.0, "lr": 5e-6, "batch_size": 8, "label_smoothing": 0.0})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
pairs = read_jsonl(I["pairs"])
val = read_jsonl(I["pairs_val"], limit=200)
fmt = I.get("model_format", "atelier")
steps_total = max(1, int(len(pairs) * float(P["epochs"]) / int(P["batch_size"])))

if fmt == "hf":
    # A HuggingFace model: TRL's DPOTrainer with a LoRA adapter. The reference is the same
    # weights with the adapter switched off, so no second copy is loaded. A fine-tuning
    # adapter the model already carries is merged into a starting copy first.
    from datasets import Dataset
    from transformers import AutoModelForCausalLM
    from trl import DPOConfig, DPOTrainer

    from atelier_nlp import hf

    system = I.get("system")
    start = I["policy"]
    if I.get("adapter"):
        progress(1, "merging the fine-tuning adapter into a starting copy")
        merged = hf.load_model(I["policy"], dtype="bf16" if device == "cuda" else "fp32", device="cpu", adapter=I["adapter"])
        start = str(run_dir / "start")
        merged.save_pretrained(start)
        hf.load_tokenizer(I["policy"]).save_pretrained(start)
        del merged

    def conv(p):
        return {"prompt": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p["prompt"]}],
                "chosen": [{"role": "assistant", "content": p["chosen"]}], "rejected": [{"role": "assistant", "content": p["rejected"]}]}

    dtype = "bf16" if device == "cuda" else "fp32"
    per_device = min(4, int(P["batch_size"]))
    grad_accum = max(1, int(P["batch_size"]) // per_device)
    total = max(1, int(math.ceil(len(pairs) / (per_device * grad_accum)) * float(P["epochs"])))
    progress(2, f"{len(pairs):,} pairs · beta {P['beta']} · {total} steps · LoRA on {I.get('model_label', I['policy'])} · {device}")
    tok = hf.load_tokenizer(start)
    model = AutoModelForCausalLM.from_pretrained(start, torch_dtype=hf.torch_dtype(dtype), local_files_only=True)
    cb = hf.ProgressCallback(total, "DPO")
    cfg = DPOConfig(
        output_dir=str(run_dir / "trainer"),
        beta=float(P["beta"]),
        label_smoothing=float(P["label_smoothing"]),
        num_train_epochs=float(P["epochs"]),
        learning_rate=float(P["lr"]),
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=min(20, total // 10),
        max_length=1024,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=max(10, total // 8),
        save_strategy="no",
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        report_to="none",
        seed=1,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list([conv(p) for p in pairs]), eval_dataset=Dataset.from_list([conv(p) for p in val]), processing_class=tok, peft_config=hf.lora_config(), callbacks=[cb.callback])
    t0 = time.time()
    trainer.train()
    trainer.evaluate()  # logged through the callback like the evaluations during training
    elapsed = time.time() - t0
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    logs = cb.history
    # TRL's names, in the shape the charts below use (its log-probabilities are summed over the answer)
    h = []
    for x in logs:
        row = {"step": x.get("step", 0)}
        if "loss" in x:
            row["loss"] = x["loss"]
        if "logps/chosen" in x:
            row["chosen_logp"], row["rejected_logp"] = x["logps/chosen"], x.get("logps/rejected")
        if "eval_rewards/accuracies" in x:
            row["eval_accuracy"], row["eval_margin"] = x["eval_rewards/accuracies"], x.get("eval_rewards/margins")
        if len(row) > 1:
            h.append(row)
    res = {"history": h, "steps": total, "elapsed_sec": elapsed}
    dpo_model, dpo_adapter = start, str(adapter_dir)
else:
    from atelier_mini.dpo import dpo
    from atelier_mini.model import MiniLM
    from atelier_mini.tok import MiniTokenizer

    model, ck = MiniLM.load(I["policy"], device)
    reference, _ = MiniLM.load(I["reference"], device)
    tok = MiniTokenizer.load(I["tokenizer"])
    system = I.get("system") or ck.get("system")
    progress(2, f"{len(pairs):,} pairs · beta {P['beta']} · {steps_total} steps")

    res = dpo(model, reference, tok, pairs, run_dir, val, beta=float(P["beta"]), epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), label_smoothing=float(P["label_smoothing"]), system=system, device=device,
              on_log=lambda r: progress(100 * r["step"] / steps_total, f"step {r['step']}/{steps_total}" + (f" · pair accuracy {r['eval_accuracy']:.0%}" if "eval_accuracy" in r else f" · loss {r.get('loss', 0):.3f}"), step=r["step"], **curves(r)))
    model.save(run_dir / "model.pt", {"dpo": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["policy"], "beta": float(P["beta"])})
    h = res["history"]
    dpo_model, dpo_adapter = str(run_dir / "model.pt"), None
evals = [x for x in h if "eval_accuracy" in x]

R = Result()
R.metric("eval_accuracy", "Pair accuracy (held out)", evals[-1]["eval_accuracy"] if evals else None, "pct", "kept", help=f"started at {evals[0]['eval_accuracy']:.0%}" if evals else None)
R.metric("eval_margin", "Margin against the reference", evals[-1]["eval_margin"] if evals else None, "num", "hold")
R.metric("steps", "Optimizer steps", res["steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("accuracy", "How often the policy prefers the chosen answer", [{"step": x["step"], "accuracy": x["eval_accuracy"]} for x in evals], "step", [{"key": "accuracy", "label": "Pair accuracy", "color": "kept"}], "line", y_domain=[0, 1])
R.chart("margin", "Margin", [{"step": x["step"], "margin": x["eval_margin"]} for x in evals], "step", [{"key": "margin", "label": "Chosen − rejected", "color": "hold"}], "line", note="A margin that keeps climbing after accuracy has saturated means the model is pushing the rejected answers down rather than the chosen ones up.")
R.chart("logp", "Per-token log-probability during training" if fmt != "hf" else "Log-probability during training (summed over the answer)", [{"step": x["step"], "chosen": x.get("chosen_logp"), "rejected": x.get("rejected_logp")} for x in h if "chosen_logp" in x], "step", [{"key": "chosen", "label": "Chosen", "color": "kept"}, {"key": "rejected", "label": "Rejected", "color": "dup"}], "line", note="If both lines fall together, the policy is collapsing: it has learned to say less rather than to choose better.")
if dpo_adapter:
    R.note(f"DPO with TRL as a LoRA adapter on {I.get('model_label', 'the HuggingFace model')}, saved in {Path(dpo_adapter).name}/; the reference was the same weights with the adapter off. Step 4 loads the model with the adapter merged in.")
else:
    R.artifact(run_dir / "model.pt", "model.pt (after DPO)")
# for a HuggingFace model, dpo_model is the starting weights and dpo_adapter holds what was learned
R.output("dpo_model", dpo_model).output("dpo_adapter", dpo_adapter).output("beta", float(P["beta"]))
for k in ("policy", "tokenizer", "lang", "system", "sft_accuracy", "model_format", "adapter", "model_label", "data_source", "eval_questions"):
    if k in I:
        R.output(k, I[k])
R.save()
