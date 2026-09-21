"""Step 3 — fine-tune on the attempts the model got right."""
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, curves, read_jsonl

parse_args()
P = params({"epochs": 3.0, "lr": 1.5e-4, "batch_size": 24, "warmup": 20})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
rows = read_jsonl(I["rft_train"])
val = read_jsonl(I["rft_val"], limit=200)
fmt = I.get("model_format", "atelier")
steps_total = max(1, int(len(rows) * float(P["epochs"]) / int(P["batch_size"])))

if fmt == "hf":
    # A HuggingFace model: a new LoRA adapter trained with TRL on the same self-written
    # examples. If the model already carries a fine-tuning adapter, it is merged into a
    # starting copy first, and the new adapter sits on top of that copy.
    from atelier_nlp import hf

    system = I.get("system")
    start = I["model"]
    if I.get("adapter"):
        progress(1, "merging the fine-tuning adapter into a starting copy")
        merged = hf.load_model(I["model"], dtype="bf16" if device == "cuda" else "fp32", device="cpu", adapter=I["adapter"])
        start = str(run_dir / "start")
        merged.save_pretrained(start)
        hf.load_tokenizer(I["model"]).save_pretrained(start)
        del merged

    def chat(r):
        return {"messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["target"]}]}

    per_device = min(4, int(P["batch_size"]))
    progress(2, f"{len(rows):,} self-written examples · LoRA on {I.get('model_label', I['model'])} · {device}")
    res = hf.sft_train(Path(start), [chat(r) for r in rows], run_dir, [chat(r) for r in val], method="lora", epochs=float(P["epochs"]), lr=float(P["lr"]), batch_size=per_device, grad_accum=max(1, int(P["batch_size"]) // per_device), warmup_ratio=min(0.3, int(P["warmup"]) / steps_total), dtype="bf16" if device == "cuda" else "fp32", gradient_checkpointing=device == "cuda", label="fine-tuning")
    hist = hf.history_chart(res["history"], ["loss", "eval_loss"])
    res["steps"] = res["total_steps"]
    rft_model, rft_adapter = start, res["save_dir"]
else:
    from atelier_mini.model import MiniLM
    from atelier_mini.tok import MiniTokenizer
    from atelier_mini.train import sft

    model, ck = MiniLM.load(I["model"], device)
    tok = MiniTokenizer.load(I["tokenizer"])
    system = I.get("system") or ck.get("system")
    progress(2, f"{len(rows):,} self-written examples · {steps_total} steps · {device}")

    res = sft(model, tok, rows, run_dir, val, epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), warmup=int(P["warmup"]), system=system, device=device,
              on_log=lambda r: progress(100 * r["step"] / steps_total, f"step {r['step']}/{steps_total}" + (f" · val {r['eval_loss']:.3f}" if "eval_loss" in r else f" · loss {r.get('loss', 0):.3f}"), step=r["step"], **curves(r)))
    model.save(run_dir / "model.pt", {"rft": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["model"]})
    hist = res["history"]
    rft_model, rft_adapter = str(run_dir / "model.pt"), None
final_eval = next((h["eval_loss"] for h in reversed(hist) if "eval_loss" in h), None)

R = Result()
R.metric("eval_loss", "Validation loss", final_eval, "num", "hold")
R.metric("examples", "Self-written examples", len(rows), "int", "kept")
R.metric("steps", "Optimizer steps", res["steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("loss", "Loss", [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")} for h in hist], "step", [{"key": "loss", "label": "Train", "color": "kept"}, {"key": "eval_loss", "label": "Validation", "color": "hold"}], "line", note="The validation targets are reference solutions, not the model's, so this loss can rise while accuracy improves — the model is learning its own style of working, not the reference's.")
if rft_adapter:
    R.note(f"A LoRA adapter on {I.get('model_label', 'the HuggingFace model')}, saved in {Path(rft_adapter).name}/; step 4 loads the model with it merged in.")
else:
    R.artifact(run_dir / "model.pt", "model.pt (after rejection-sampled fine-tuning)")
# for a HuggingFace model, rft_model is the starting weights and rft_adapter holds what was learned
R.output("rft_model", rft_model).output("rft_adapter", rft_adapter)
for k in ("model", "tokenizer", "lang", "system", "coverage", "model_format", "adapter", "model_label", "data_source", "qa_val"):
    if k in I:
        R.output(k, I[k])
R.save()
