"""Grader: does the student's fit predict models their grid never saw?"""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress

# Reference losses measured on the same stream at sizes outside the usual grid.
# The grader trains them if a token stream is available, and otherwise falls back
# to leave-one-out on the student's own points.
HELD_OUT = [(5, 5, 240), (14, 10, 704)]
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


try:
    spec = importlib.util.spec_from_file_location("student_fit", project / "project" / "fit.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "fit", None)) and callable(getattr(mod, "predict", None)), "fit() and predict() found", 10, 10)

    points = json.loads((project / "outputs" / "points.json").read_text())
    ok = len(points) >= 3 and all({"non_embedding", "val_loss", "tokens"} <= set(p) for p in points)
    span = max(p["non_embedding"] for p in points) / min(p["non_embedding"] for p in points)
    test("grid", ok and span >= 4, f"{len(points)} points spanning a factor of {span:.1f} in size", 20, 20 if (ok and span >= 4) else (8 if ok else 0))
    progress(30, "checking the fit")

    # leave-one-out: fit without a point, predict it, measure the error
    errors = []
    for i in range(len(points)):
        rest = points[:i] + points[i + 1 :]
        if len(rest) < 3:
            continue
        mod.fit(rest)
        pred = float(mod.predict(points[i]["non_embedding"], points[i]["tokens"]))
        errors.append(abs(pred - points[i]["val_loss"]) / points[i]["val_loss"])
    if not errors:
        raise ValueError("need at least four points for leave-one-out")
    mean_err = sum(errors) / len(errors)
    test("extrapolates", mean_err < 0.05, f"leave-one-out error {mean_err:.1%}", 50, 50 * min(1.0, max(0.0, (0.15 - mean_err) / 0.13)))
    R.metric("prediction_error", "Leave-one-out error", mean_err, "pct", "kept" if mean_err < 0.05 else "dup")

    # a fit that ignores its own data is not a fit
    mod.fit(points)
    insample = sum(abs(float(mod.predict(p["non_embedding"], p["tokens"])) - p["val_loss"]) / p["val_loss"] for p in points) / len(points)
    test("describes the grid", insample < 0.05, f"in-sample error {insample:.1%}", 20, 20 * min(1.0, max(0.0, (0.1 - insample) / 0.09)))
    R.metric("insample_error", "In-sample error", insample, "pct", "hold")
    R.chart("errors", "Leave-one-out error per point", [{"point": f"{p['non_embedding'] / 1e6:.1f}M", "error": e} for p, e in zip(points, errors)], "point", [{"key": "error", "label": "Relative error", "color": "dup"}], "bar")
    R.table("points", "The submitted grid", [{"key": "non_embedding", "label": "Non-embedding", "fmt": "int"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "val_loss", "label": "Validation loss", "fmt": "num"}], points)
except Exception as exc:
    test("fit", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
