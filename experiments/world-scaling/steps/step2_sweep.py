"""Step 2 — train every point, holding everything but size fixed."""
import json
import math
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.train import pretrain

parse_args()
P = params({"lr_scaling": "sqrt", "base_lr": 1e-3, "warmup_frac": 0.05, "eval_every": 100, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
plan = json.loads(Path(I["plan"]).read_text())
meta = json.loads(Path(I["meta"]).read_text())
device = "cuda" if torch.cuda.is_available() else "cpu"
train = TokenStream(I["train_bin"], meta["dtype"])
val = TokenStream(I["val_bin"], meta["dtype"])
points = plan["points"]
total_iters = sum(p["iters"] for p in points)
done_iters = 0
results, curves = [], []

for pi, point in enumerate(points):
    torch.manual_seed(int(P["seed"]))
    cfg = MiniConfig(**point["config"])
    model = MiniLM(cfg).to(device)
    lr = float(P["base_lr"]) * (math.sqrt(288 / cfg.n_embd) if P["lr_scaling"] == "sqrt" else 1.0)
    out = run_dir / point["key"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {point['label']} · {model.num_params():,} parameters · lr {lr:.2e} · {point['iters']} steps", flush=True)

    def on_log(row, pi=pi, point=point):
        global done_iters
        if "val_loss" in row:
            progress(100 * (done_iters + row["step"]) / total_iters, f"{point['label']} · step {row['step']}/{point['iters']} · val {row['val_loss']:.3f}", step=done_iters + row["step"], val_loss=row["val_loss"])

    res = pretrain(model, train, val, out, max_iters=point["iters"], batch_size=plan["batch_size"], block_size=cfg.block_size, lr=lr, warmup=max(10, int(point["iters"] * float(P["warmup_frac"]))), eval_every=int(P["eval_every"]), eval_iters=20, device=device, on_log=on_log)
    done_iters += point["iters"]
    results.append({**{k: point[k] for k in ("key", "label", "params", "non_embedding", "tokens", "iters")}, "val_loss": res["best_val_loss"], "lr": lr, "elapsed": res["elapsed_sec"], "checkpoint": res["checkpoint"]})
    for h in res["history"]:
        curves.append({"step": h["step"], "tokens": h["tokens"], point["label"]: h["val_loss"]})
    del model
    torch.cuda.empty_cache()

(run_dir / "points.json").write_text(json.dumps(results, indent=2))
merged = {}
for row in curves:
    m = merged.setdefault(row["tokens"], {"tokens": row["tokens"]})
    m.update({k: v for k, v in row.items() if k not in ("step", "tokens")})

R = Result()
R.metric("trained", "Models trained", len(results), "int", "kept")
R.metric("best_loss", "Best validation loss", min(r["val_loss"] for r in results), "num", "hold", help=max(results, key=lambda r: -r["val_loss"])["label"])
R.metric("total_time", "Total training time", sum(r["elapsed"] for r in results) * 1000, "ms", "sky")
R.chart("loss_vs_params", "Loss against size", [{"params": r["non_embedding"], "val_loss": r["val_loss"]} for r in results], "params", [{"key": "val_loss", "label": "Validation loss", "color": "kept"}], "line", x_log=True, y_log=True, note="On log axes a power law is a straight line. Curvature at the small end is normal.")
R.chart("curves", "Every run, loss against tokens seen", [merged[k] for k in sorted(merged)], "tokens", [{"key": r["label"], "label": r["label"]} for r in results], "line", x_log=True, y_log=True, note="Larger models start worse and cross over. Where they cross is the size the token budget can actually support.")
R.table("points", "The grid", [{"key": "label", "label": "Shape"}, {"key": "params", "label": "Parameters", "fmt": "int"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "lr", "label": "Learning rate", "fmt": "num"}, {"key": "val_loss", "label": "Validation loss", "fmt": "num"}, {"key": "elapsed", "label": "Seconds", "fmt": "num"}], results)
R.artifact(run_dir / "points.json", "points.json")
R.output("points", str(run_dir / "points.json")).output("plan", I["plan"]).output("train_bin", I["train_bin"]).output("val_bin", I["val_bin"]).output("meta", I["meta"]).output("vocab_size", I["vocab_size"])
R.save()
