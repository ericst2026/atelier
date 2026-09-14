"""Step 3 — fine-tune on the attempts the model got right."""
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import sft

parse_args()
P = params({"epochs": 3.0, "lr": 1.5e-4, "batch_size": 24, "warmup": 20})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
rows = read_jsonl(I["rft_train"])
val = read_jsonl(I["rft_val"], limit=200)
model, ck = MiniLM.load(I["model"], device)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or ck.get("system")
steps_total = max(1, int(len(rows) * float(P["epochs"]) / int(P["batch_size"])))
progress(2, f"{len(rows):,} self-written examples · {steps_total} steps · {device}")

res = sft(model, tok, rows, run_dir, val, epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), warmup=int(P["warmup"]), system=system, device=device,
          on_log=lambda r: progress(100 * r["step"] / steps_total, f"step {r['step']}/{steps_total}" + (f" · val {r['eval_loss']:.3f}" if "eval_loss" in r else f" · loss {r.get('loss', 0):.3f}"), step=r["step"], **{k: v for k, v in r.items() if k in ("loss", "eval_loss")}))
model.save(run_dir / "model.pt", {"rft": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["model"]})
hist = res["history"]
final_eval = next((h["eval_loss"] for h in reversed(hist) if "eval_loss" in h), None)

R = Result()
R.metric("eval_loss", "Validation loss", final_eval, "num", "hold")
R.metric("examples", "Self-written examples", len(rows), "int", "kept")
R.metric("steps", "Optimizer steps", res["steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("loss", "Loss", [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")} for h in hist], "step", [{"key": "loss", "label": "Train", "color": "kept"}, {"key": "eval_loss", "label": "Validation", "color": "hold"}], "line", note="The validation targets are the generator's solutions, not the model's, so this loss can rise while accuracy improves — the model is learning its own style of working, not the generator's.")
R.artifact(run_dir / "model.pt", "model.pt (after rejection-sampled fine-tuning)")
R.output("rft_model", str(run_dir / "model.pt"))
for k in ("model", "tokenizer", "lang", "system", "coverage"):
    if k in I:
        R.output(k, I[k])
R.save()
