"""Step 3 — supervised fine-tuning with the question masked out of the loss."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, curves, read_jsonl

parse_args()
P = params({"epochs": 2.0, "lr": 2e-4, "batch_size": 24, "warmup": 20, "weight_decay": 0.1, "use_system": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.loads(Path(I["sft_meta"]).read_text())
rows = read_jsonl(I["sft_train"])
val = read_jsonl(I["sft_val"], limit=256)
system = meta["system"] if bool(P["use_system"]) else None
fmt = I.get("model_format", "atelier")
steps_total = max(1, int(len(rows) * float(P["epochs"]) / int(P["batch_size"])))


def on_log(row):
    msg = f"step {row['step']}/{steps_total}"
    if "eval_loss" in row:
        msg += f" · val {row['eval_loss']:.3f}"
    elif "loss" in row:
        msg += f" · loss {row['loss']:.3f}"
    progress(100 * row["step"] / steps_total, msg, step=row["step"], **curves(row))


if fmt == "hf":
    # A HuggingFace model: a LoRA adapter trained with TRL on the same demonstrations,
    # as a chat (system, question, target). The question is masked the same way.
    from atelier_nlp import hf

    def chat(r):
        return {"messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["target"]}]}

    per_device = min(4, int(P["batch_size"]))
    progress(2, f"{len(rows):,} demonstrations · LoRA on {I.get('model_label', I['base_model'])} · {device}")
    res = hf.sft_train(Path(I["base_model"]), [chat(r) for r in rows], run_dir, [chat(r) for r in val], method="lora", epochs=float(P["epochs"]), lr=float(P["lr"]), batch_size=per_device, grad_accum=max(1, int(P["batch_size"]) // per_device), warmup_ratio=min(0.3, int(P["warmup"]) / steps_total), dtype="bf16" if device == "cuda" else "fp32", gradient_checkpointing=device == "cuda", label="fine-tuning")
    hist = hf.history_chart(res["history"], ["loss", "eval_loss"])
    res["steps"] = res["total_steps"]
    model_path, adapter = I["base_model"], res["save_dir"]
else:
    from atelier_mini.model import MiniLM
    from atelier_mini.tok import MiniTokenizer
    from atelier_mini.train import sft

    model, ck = MiniLM.load(I["base_model"], device)
    tok = MiniTokenizer.load(I["tokenizer"])
    progress(2, f"{len(rows):,} demonstrations · {model.num_params():,} parameters · {device}")
    res = sft(model, tok, rows, run_dir, val, epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), warmup=int(P["warmup"]), weight_decay=float(P["weight_decay"]), system=system, device=device, on_log=on_log)
    model.save(run_dir / "model.pt", {"sft": True, "tokenizer": str(I["tokenizer"]), "lang": meta["lang"], "system": system})
    hist = res["history"]
    model_path, adapter = str(run_dir / "model.pt"), None
final_eval = next((h["eval_loss"] for h in reversed(hist) if "eval_loss" in h), None)
first_eval = next((h["eval_loss"] for h in hist if "eval_loss" in h), None)

R = Result()
R.metric("eval_loss", "Validation loss", final_eval, "num", "hold", help=f"started at {first_eval:.3f}" if first_eval else None)
R.metric("steps", "Optimizer steps", res["steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.metric("examples_seen", "Demonstrations seen", res["steps"] * int(P["batch_size"]), "int", "raw")
R.chart("loss", "Loss on the answer tokens", [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")} for h in hist], "step", [{"key": "loss", "label": "Train", "color": "kept"}, {"key": "eval_loss", "label": "Validation", "color": "hold"}], "line", note="Only the target tokens count. If the validation line turns upward, stop earlier — the model is memorising these demonstrations.")
if adapter:
    R.note(f"A LoRA adapter on {I.get('model_label', 'the prepared model')}, saved in {Path(adapter).name}/; later steps and experiments load the base with it merged in.")
else:
    R.artifact(run_dir / "model.pt", "model.pt (fine-tuned)")
# for a HuggingFace base, sft_model is the base and `adapter` holds what was learned
R.output("sft_model", model_path).output("adapter", adapter).output("model_format", fmt).output("eval_loss", final_eval).output("system", system)
for k in ("base_model", "tokenizer", "sft_meta", "sft_val", "lang", "baseline_accuracy", "baseline_format", "model_label"):
    if k in I:
        R.output(k, I[k])
R.save()
