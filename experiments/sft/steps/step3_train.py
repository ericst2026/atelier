"""Step 3 — supervised fine-tuning with TRL."""
import json
import os
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"epochs": 1.0, "lr": 2e-4, "batch_size": 4, "grad_accum": 4, "warmup_ratio": 0.03, "packing": False})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
setup = json.loads(Path(I["setup"]).read_text())
train = hf.read_jsonl(I["train"])
val = hf.read_jsonl(I["val"], limit=200)
progress(2, f"{len(train)} training examples · {setup['method']} on {setup['base_model_name']}")
res = hf.sft_train(Path(setup["base_model"]), [{"messages": r["messages"]} for r in train], run_dir, [{"messages": r["messages"]} for r in val], method=setup["method"], lora=setup["lora"] if setup["method"] == "lora" else None, epochs=float(P["epochs"]), lr=float(P["lr"]), batch_size=int(P["batch_size"]), grad_accum=int(P["grad_accum"]), max_length=setup["max_length"], packing=bool(P["packing"]), warmup_ratio=float(P["warmup_ratio"]), dtype=setup["dtype"], gradient_checkpointing=setup["gradient_checkpointing"])
hist_rows = hf.history_chart(res["history"], ["loss", "eval_loss", "learning_rate", "grad_norm"])
final_loss = next((h["loss"] for h in reversed(res["history"]) if "loss" in h), None)
eval_loss = res["final_eval"].get("eval_loss")

R = Result()
R.metric("train_loss", "Final training loss", final_loss, "num", "kept")
R.metric("eval_loss", "Validation loss", eval_loss, "num", "hold")
R.metric("steps", "Optimizer steps", res["total_steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("loss", "Loss", [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")} for h in hist_rows], "step", [{"key": "loss", "label": "Train", "color": "kept"}, {"key": "eval_loss", "label": "Validation", "color": "hold"}], "line")
R.chart("lr", "Learning rate", [{"step": h["step"], "lr": h["learning_rate"]} for h in hist_rows if "learning_rate" in h], "step", [{"key": "lr", "label": "lr", "color": "sky"}], "line")
R.chart("gn", "Gradient norm", [{"step": h["step"], "grad_norm": h["grad_norm"]} for h in hist_rows if "grad_norm" in h], "step", [{"key": "grad_norm", "label": "grad norm", "color": "raw"}], "line")
R.artifact(Path(res["save_dir"]), "trained weights")
R.output("model_dir", res["save_dir"]).output("method", res["method"]).output("base_model", setup["base_model"]).output("val", I["val"]).output("train", I["train"]).output("max_length", setup["max_length"])
R.save()
