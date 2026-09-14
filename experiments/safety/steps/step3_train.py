"""Step 3 — teach the boundary from both sides."""
import json
import os
import random
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_safety import is_refusal, refusal_text  # noqa: E402

parse_args()
P = params({"n_refusals": 1500, "helpful_ratio": 2.0, "epochs": 2.0, "lr": 2e-5, "batch_size": 8, "method": "lora"})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
rng = random.Random(3)
rows = hf.read_jsonl(I["prompts"])
unsafe_prompts = [r["prompt"] for r in rows if r["expected"] == "refuse"]

# refusal demonstrations: the prompt, and a refusal that says why and offers an alternative
try:
    pku = hf.dataset_split("safety", "pku-saferlhf", "train", limit=int(P["n_refusals"]) * 8)
except Exception:
    pku = []
refusals = []
for r in pku:
    if not r.get("prompt"):
        continue
    refusals.append({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": refusal_text(r["prompt"])}]})
    if len(refusals) >= int(P["n_refusals"]):
        break
for p in unsafe_prompts:
    if len(refusals) >= int(P["n_refusals"]):
        break
    refusals.append({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": refusal_text(p)}]})

# helpful demonstrations, so the model learns where the boundary is rather than to decline
n_helpful = int(len(refusals) * float(P["helpful_ratio"]))
helpful = []
if n_helpful:
    for r in hf.dataset_split("sft", "no-robots", "train", limit=n_helpful * 3):
        msgs = r.get("messages") or []
        if len(msgs) >= 2 and msgs[0].get("role") == "user" and msgs[1].get("role") == "assistant":
            helpful.append({"messages": [{"role": "user", "content": msgs[0]["content"]}, {"role": "assistant", "content": msgs[1]["content"]}]})
        if len(helpful) >= n_helpful:
            break
train = refusals + helpful
rng.shuffle(train)
val = train[: max(40, len(train) // 20)]
train = train[len(val):]
if len(train) < 40:
    raise SystemExit("Not enough demonstrations. Check that the safety and sft dataset groups are installed.")
progress(15, f"{len(refusals)} refusals and {len(helpful)} helpful demonstrations")

mp = hf.model_path(I["model"])
res = hf.sft_train(mp, train, run_dir, val, method=P["method"], lora={"r": 16, "alpha": 32} if P["method"] == "lora" else None,
                   epochs=float(P["epochs"]), lr=float(P["lr"]), batch_size=int(P["batch_size"]), grad_accum=2, max_length=1024, label="safety")
hist_rows = hf.history_chart(res["history"], ["loss", "eval_loss"])
final_eval = res["final_eval"].get("eval_loss")

R = Result()
R.metric("eval_loss", "Validation loss", final_eval, "num", "hold")
R.metric("refusals", "Refusal demonstrations", len(refusals), "int", "dup")
R.metric("helpful", "Helpful demonstrations", len(helpful), "int", "kept", help=f"{float(P['helpful_ratio']):.1f} per refusal")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("mix", "The training mix", [{"kind": "refusals", "count": len(refusals)}, {"kind": "helpful", "count": len(helpful)}], "kind", [{"key": "count", "label": "Demonstrations", "color": "kept"}], "bar", note="With no helpful demonstrations the model learns that declining is always safe, and the next step will show exactly what that costs.")
R.chart("loss", "Loss", [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss")} for h in hist_rows], "step", [{"key": "loss", "label": "Train", "color": "kept"}, {"key": "eval_loss", "label": "Validation", "color": "hold"}], "line")
R.table("samples", "A refusal demonstration", [{"key": "prompt", "label": "Prompt"}, {"key": "target", "label": "Trained response"}], [{"prompt": r["messages"][0]["content"][:200], "target": r["messages"][1]["content"][:260]} for r in refusals[:6]])
R.artifact(Path(res["save_dir"]), "trained weights")
R.output("trained_model", res["save_dir"]).output("method", P["method"]).output("base_model", str(mp)).output("helpful_ratio", float(P["helpful_ratio"]))
for k in ("prompts", "model", "system_prompt", "violation", "over_refusal", "baseline_safety"):
    if k in I:
        R.output(k, I[k])
R.save()
