"""Step 3 — DPO."""
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.dpo import dpo
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer

parse_args()
P = params({"beta": 0.1, "epochs": 1.0, "lr": 5e-6, "batch_size": 8, "label_smoothing": 0.0})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
pairs = read_jsonl(I["pairs"])
val = read_jsonl(I["pairs_val"], limit=200)
model, ck = MiniLM.load(I["policy"], device)
reference, _ = MiniLM.load(I["reference"], device)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or ck.get("system")
steps_total = max(1, int(len(pairs) * float(P["epochs"]) / int(P["batch_size"])))
progress(2, f"{len(pairs):,} pairs · beta {P['beta']} · {steps_total} steps")

res = dpo(model, reference, tok, pairs, run_dir, val, beta=float(P["beta"]), epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), label_smoothing=float(P["label_smoothing"]), system=system, device=device,
          on_log=lambda r: progress(100 * r["step"] / steps_total, f"step {r['step']}/{steps_total}" + (f" · pair accuracy {r['eval_accuracy']:.0%}" if "eval_accuracy" in r else f" · loss {r.get('loss', 0):.3f}"), step=r["step"], **{k: v for k, v in r.items() if k in ("loss", "accuracy", "margin", "eval_accuracy", "eval_margin", "chosen_logp", "rejected_logp")}))
model.save(run_dir / "model.pt", {"dpo": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["policy"], "beta": float(P["beta"])})
h = res["history"]
evals = [x for x in h if "eval_accuracy" in x]

R = Result()
R.metric("eval_accuracy", "Pair accuracy (held out)", evals[-1]["eval_accuracy"] if evals else None, "pct", "kept", help=f"started at {evals[0]['eval_accuracy']:.0%}" if evals else None)
R.metric("eval_margin", "Margin against the reference", evals[-1]["eval_margin"] if evals else None, "num", "hold")
R.metric("steps", "Optimizer steps", res["steps"], "int", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("accuracy", "How often the policy prefers the chosen answer", [{"step": x["step"], "accuracy": x["eval_accuracy"]} for x in evals], "step", [{"key": "accuracy", "label": "Pair accuracy", "color": "kept"}], "line", y_domain=[0, 1])
R.chart("margin", "Margin", [{"step": x["step"], "margin": x["eval_margin"]} for x in evals], "step", [{"key": "margin", "label": "Chosen − rejected", "color": "hold"}], "line", note="A margin that keeps climbing after accuracy has saturated means the model is pushing the rejected answers down rather than the chosen ones up.")
R.chart("logp", "Per-token log-probability during training", [{"step": x["step"], "chosen": x.get("chosen_logp"), "rejected": x.get("rejected_logp")} for x in h if "chosen_logp" in x], "step", [{"key": "chosen", "label": "Chosen", "color": "kept"}, {"key": "rejected", "label": "Rejected", "color": "dup"}], "line", note="If both lines fall together, the policy is collapsing: it has learned to say less rather than to choose better.")
R.artifact(run_dir / "model.pt", "model.pt (after DPO)")
R.output("dpo_model", str(run_dir / "model.pt")).output("beta", float(P["beta"]))
for k in ("policy", "tokenizer", "lang", "system", "sft_accuracy"):
    if k in I:
        R.output(k, I[k])
R.save()
