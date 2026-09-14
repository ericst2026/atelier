"""Step 4 — train the predicted model and compare."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.train import pretrain

parse_args()
P = params({"max_iters": 0})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
fit = json.loads(Path(I["fit"]).read_text())
plan = json.loads(Path(I["plan"]).read_text())
meta = json.loads(Path(I["meta"]).read_text())
points = json.loads(Path(I["points"]).read_text())
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = MiniConfig(**fit["config"])
iters = int(P["max_iters"]) or max(50, fit["tokens"] // (plan["batch_size"] * cfg.block_size))
torch.manual_seed(1)
model = MiniLM(cfg).to(device)
print(f"[verify] {model.num_params():,} parameters · {iters} steps · predicted loss {fit['predicted_loss']:.4f}", flush=True)

res = pretrain(model, TokenStream(I["train_bin"], meta["dtype"]), TokenStream(I["val_bin"], meta["dtype"]), run_dir,
               max_iters=iters, batch_size=plan["batch_size"], block_size=cfg.block_size, lr=1e-3 * (288 / cfg.n_embd) ** 0.5, warmup=max(10, iters // 20), device=device,
               on_log=lambda r: progress(100 * r["step"] / iters, f"step {r['step']}/{iters}" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else ""), step=r["step"], **{k: v for k, v in r.items() if k in ("val_loss", "loss")}))
measured = res["best_val_loss"]
predicted = fit["predicted_loss"]
error = (measured - predicted) / predicted

rows = [{"label": p["label"], "params": p["non_embedding"], "measured": p["val_loss"], "kind": "grid"} for p in sorted(points, key=lambda p: p["non_embedding"])]
rows.append({"label": "prediction", "params": fit["target"]["non_embedding"], "measured": measured, "predicted": predicted, "kind": "held out"})

R = Result()
R.metric("measured", "Measured loss", measured, "num", "kept")
R.metric("predicted", "Predicted loss", predicted, "num", "hold")
R.metric("error", "Relative error", abs(error), "pct", "kept" if abs(error) < 0.05 else "dup", help=f"{'above' if error > 0 else 'below'} the prediction")
R.metric("params", "Parameters", model.num_params(), "int", "sky")
R.chart("compare", "Prediction against measurement", [{"which": "predicted", "loss": predicted}, {"which": "measured", "loss": measured}], "which", [{"key": "loss", "label": "Validation loss", "color": "kept"}], "bar")
R.chart("all", "The whole curve, with the held-out point", [{"params": r["params"], "measured": r["measured"], "predicted": r.get("predicted")} for r in rows], "params", [{"key": "measured", "label": "Measured", "color": "kept"}, {"key": "predicted", "label": "Predicted", "color": "raw"}], "line", x_log=True, y_log=True)
R.chart("loss_curve", "The verification run", [{"tokens": h["tokens"], "val_loss": h["val_loss"]} for h in res["history"] if h["step"]], "tokens", [{"key": "val_loss", "label": "Validation loss", "color": "hold"}], "line", x_log=True, y_log=True)
R.table("all", "Everything", [{"key": "label", "label": "Shape"}, {"key": "params", "label": "Non-embedding", "fmt": "int"}, {"key": "measured", "label": "Measured", "fmt": "num"}, {"key": "predicted", "label": "Predicted", "fmt": "num"}, {"key": "kind", "label": "In the fit?"}], rows)
if abs(error) > 0.1:
    R.note("More than ten percent out. The usual causes, in order: the verification model saw fewer tokens per parameter than the grid did, the learning rate rule does not hold at this width, or the grid spanned too small a range of sizes to extrapolate from.")
R.artifact(Path(res["checkpoint"]), "model.pt")
R.output("prediction_error", abs(error)).output("measured_loss", measured).output("model", res["checkpoint"])
R.save()
