"""Step 3 — fit L(N) = E + A·N^(−α) and extrapolate one step past the grid."""
import json
import math
import os
import sys
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args
from atelier_mini.model import MiniConfig, estimate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_grid import GRID  # noqa: E402

parse_args()
P = params({"exclude_smallest": False, "predict_size": "n5"})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
points = json.loads(Path(I["points"]).read_text())
plan = json.loads(Path(I["plan"]).read_text())
used = sorted(points, key=lambda p: p["non_embedding"])
if bool(P["exclude_smallest"]) and len(used) > 3:
    used = used[1:]
if len(used) < 3:
    raise SystemExit("A fit needs at least three points; train more of the grid first.")


def fit_power(pts, E: float):
    """log(L − E) = log A − α·log N, by ordinary least squares."""
    xs, ys = [], []
    for p in pts:
        r = p["val_loss"] - E
        if r <= 1e-6:
            return None
        xs.append(math.log(p["non_embedding"]))
        ys.append(math.log(r))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    tot = sum((y - my) ** 2 for y in ys) or 1e-9
    return {"E": E, "A": math.exp(intercept), "alpha": -slope, "r2": 1 - resid / tot}


# E is the irreducible loss; scan for the value that makes the power law straightest
best = None
lo_loss = min(p["val_loss"] for p in used)
for i in range(200):
    E = lo_loss * i / 200.0
    f = fit_power(used, E)
    if f and (best is None or f["r2"] > best["r2"]):
        best = f
if best is None:
    raise SystemExit("Could not fit a curve to these points — check that the losses fall with size.")


def predict(n_non_emb: float) -> float:
    return best["E"] + best["A"] * n_non_emb ** (-best["alpha"])


rows = [{"label": p["label"], "params": p["non_embedding"], "measured": p["val_loss"], "predicted": predict(p["non_embedding"]), "residual": p["val_loss"] - predict(p["non_embedding"])} for p in sorted(points, key=lambda p: p["non_embedding"])]
L, H, D, label = GRID[P["predict_size"]]
cfg = MiniConfig(vocab_size=int(I["vocab_size"]), n_layer=L, n_head=H, n_embd=D, block_size=plan["block_size"])
target = estimate(cfg)
predicted = predict(target["non_embedding"])
curve = []
lo = min(p["non_embedding"] for p in points) / 2
hi = target["non_embedding"] * 1.5
for i in range(40):
    n = lo * (hi / lo) ** (i / 39)
    curve.append({"params": n, "fit": predict(n)})
for p in points:
    curve.append({"params": p["non_embedding"], "measured": p["val_loss"], "fit": predict(p["non_embedding"])})
curve.sort(key=lambda r: r["params"])
(run_dir / "fit.json").write_text(json.dumps({"fit": best, "predict_size": P["predict_size"], "target": target, "predicted_loss": predicted, "config": cfg.to_dict(), "tokens": int(plan["tokens_per_param"] * target["total"])}, indent=2))

R = Result()
R.metric("alpha", "Exponent α", best["alpha"], "num", "kept", help="published values for language models sit near 0.07–0.1 in this parameterisation; a small grid on generated text can differ a lot")
R.metric("E", "Irreducible loss E", best["E"], "num", "hold", help="what the curve says no model of this family will beat on this data")
R.metric("r2", "Fit quality", best["r2"], "pct", "sky", help="R² of the straight line in log space")
R.metric("predicted", f"Predicted loss for {label}", predicted, "num", "raw", help=f"{target['total'] / 1e6:.1f}M parameters — not yet trained")
R.chart("fit", "The fit", curve, "params", [{"key": "fit", "label": "L(N) = E + A·N^(−α)", "color": "hold"}, {"key": "measured", "label": "Measured", "color": "kept"}], "line", x_log=True, y_log=True, note="The fitted line continues past the last measured point; the next step trains that model and checks it.")
R.chart("residuals", "Residuals", [{"label": r["label"], "residual": r["residual"]} for r in rows], "label", [{"key": "residual", "label": "Measured − predicted", "color": "dup"}], "bar", note="A point sitting above the line usually saw too few tokens for its size, not too little capacity.")
R.table("points", "Measured against fitted", [{"key": "label", "label": "Shape"}, {"key": "params", "label": "Non-embedding", "fmt": "int"}, {"key": "measured", "label": "Measured", "fmt": "num"}, {"key": "predicted", "label": "Fitted", "fmt": "num"}, {"key": "residual", "label": "Residual", "fmt": "num"}], rows)
R.artifact(run_dir / "fit.json", "fit.json")
R.output("fit", str(run_dir / "fit.json")).output("predicted_loss", predicted).output("alpha", best["alpha"])
for k in ("points", "plan", "train_bin", "val_bin", "meta", "vocab_size"):
    R.output(k, I[k])
R.save()
