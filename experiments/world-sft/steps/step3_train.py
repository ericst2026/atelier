"""Step 3 — supervised fine-tuning with the question masked out of the loss."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import sft

parse_args()
P = params({"epochs": 2.0, "lr": 2e-4, "batch_size": 24, "warmup": 20, "weight_decay": 0.1, "use_system": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.loads(Path(I["sft_meta"]).read_text())
rows = read_jsonl(I["sft_train"])
val = read_jsonl(I["sft_val"], limit=256)
model, ck = MiniLM.load(I["base_model"], device)
tok = MiniTokenizer.load(I["tokenizer"])
system = meta["system"] if bool(P["use_system"]) else None
progress(2, f"{len(rows):,} demonstrations · {model.num_params():,} parameters · {device}")

steps_total = max(1, int(len(rows) * float(P["epochs"]) / int(P["batch_size"])))


def on_log(row):
    msg = f"step {row['step']}/{steps_total}"
    if "eval_loss" in row:
        msg += f" · val {row['eval_loss']:.3f}"
    elif "loss" in row:
        msg += f" · loss {row['loss']:.3f}"
    progress(100 * row["step"] / steps_total, msg, step=row["step"], **{k: v for k, v in row.items() if k in ("loss", "eval_loss")})


res = sft(model, tok, rows, run_dir, val, epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), warmup=int(P["warmup"]), weight_decay=float(P["weight_decay"]), system=system, device=device, on_log=on_log)
model.save(run_dir / "model.pt", {"sft": True, "tokenizer": str(I["tokenizer"]), "lang": meta["lang"], "system": system})
hist = res["history"]
final_eval = next((h["eval_loss"] for h in reversed(hist) if "eval_loss" in h), None)
first_eval = next((h["eval_loss"] for h in hist if "eval_loss" in h), None)

R = Result()
R.metric("eval_loss", "Validation loss", final_eval, "num", "hold", help=f"started at {first_eval:.3f}" if first_eval else None)
R.metric("steps", "Optimizer steps", res["steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.metric("examples_seen", "Demonstrations seen", res["steps"] * int(P["batch_size"]), "int", "raw")
R.chart("loss", "Loss on the answer tokens", [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")} for h in hist], "step", [{"key": "loss", "label": "Train", "color": "kept"}, {"key": "eval_loss", "label": "Validation", "color": "hold"}], "line", note="Only the target tokens count. If the validation line turns upward, stop earlier — the model is memorising these demonstrations.")
R.artifact(run_dir / "model.pt", "model.pt (fine-tuned)")
R.output("sft_model", str(run_dir / "model.pt")).output("eval_loss", final_eval).output("system", system)
for k in ("base_model", "tokenizer", "sft_meta", "sft_val", "lang", "baseline_accuracy", "baseline_format"):
    if k in I:
        R.output(k, I[k])
R.save()
