"""Step 3 — fine-tune on the traces, with the tool's output masked out of the loss."""
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl

parse_args()
P = params({"epochs": 3.0, "lr": 2e-4, "batch_size": 24})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
rows = read_jsonl(I["traces"])
val = read_jsonl(I["traces_val"], limit=200)
fmt = I.get("model_format", "atelier")
steps_total = max(1, int(len(rows) * float(P["epochs"]) / int(P["batch_size"])))

if fmt == "hf":
    # A HuggingFace model: a LoRA adapter trained with TRL on the same traces, as a chat.
    # A fine-tuning adapter the model already carries is merged into a starting copy first.
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

    def chat(r):
        return {"messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["target"]}]}

    per_device = min(4, int(P["batch_size"]))
    progress(2, f"{len(rows):,} demonstrations · LoRA on {I.get('model_label', I['policy'])} · {device}")
    res = hf.sft_train(Path(start), [chat(r) for r in rows], run_dir, [chat(r) for r in val], method="lora", epochs=float(P["epochs"]), lr=float(P["lr"]), batch_size=per_device, grad_accum=max(1, int(P["batch_size"]) // per_device), dtype="bf16" if device == "cuda" else "fp32", gradient_checkpointing=device == "cuda", label="fine-tuning")
    hist = hf.history_chart(res["history"], ["loss", "eval_loss"])
    res["steps"] = res["total_steps"]
    tool_model, tool_adapter = start, res["save_dir"]
else:
    from atelier_mini.model import MiniLM
    from atelier_mini.tok import MiniTokenizer
    from atelier_mini.train import sft

    model, ck = MiniLM.load(I["policy"], device)
    tok = MiniTokenizer.load(I["tokenizer"])
    system = I.get("system") or ck.get("system")
    progress(2, f"{len(rows):,} demonstrations · {steps_total} steps")

    res = sft(model, tok, rows, run_dir, val, epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), system=system, device=device,
              on_log=lambda r: progress(100 * r["step"] / steps_total, f"step {r['step']}/{steps_total}" + (f" · val {r['eval_loss']:.3f}" if "eval_loss" in r else f" · loss {r.get('loss', 0):.3f}"), step=r["step"], **{k: v for k, v in r.items() if k in ("loss", "eval_loss")}))
    model.save(run_dir / "model.pt", {"tools": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["policy"]})
    hist = res["history"]
    tool_model, tool_adapter = str(run_dir / "model.pt"), None
final_eval = next((h["eval_loss"] for h in reversed(hist) if "eval_loss" in h), None)

R = Result()
R.metric("eval_loss", "Validation loss", final_eval, "num", "hold")
R.metric("steps", "Optimizer steps", res["steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "raw")
R.chart("loss", "Loss", [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")} for h in hist], "step", [{"key": "loss", "label": "Train", "color": "kept"}, {"key": "eval_loss", "label": "Validation", "color": "hold"}], "line")
if tool_adapter:
    R.note(f"A LoRA adapter on {I.get('model_label', 'the HuggingFace model')}, saved in {Path(tool_adapter).name}/. TRL's loss covers the whole conversation here, the question and the tool's bracketed result included.")
else:
    R.artifact(run_dir / "model.pt", "model.pt (tool-using)")
# for a HuggingFace model, tool_model is the starting weights and tool_adapter holds what was learned
R.output("tool_model", tool_model).output("tool_adapter", tool_adapter)
for k in ("policy", "tokenizer", "tools_config", "lang", "system", "baseline", "model_format", "adapter", "model_label", "data_source", "qa_val"):
    if k in I:
        R.output(k, I[k])
R.save()
