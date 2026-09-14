"""Grader: throughput per GPU and the loss reached, on one GPU and on four."""
import json
import os
import subprocess
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress

args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


data = os.environ.get("ATELIER_GRADER_DATA")
try:
    import torch

    cpu = os.environ.get("ATELIER_DEVICE") == "cpu" or not torch.cuda.is_available()
except ImportError:
    cpu = os.environ.get("ATELIER_DEVICE") == "cpu"
# on a CPU-only node the throughput bar (set for GPUs) is not scored; the rest is scaled to 100
possible = 60 if (data and cpu) else 100
entry = project / "project" / "train_ddp.py"
rows = []
try:
    test("interface", entry.exists(), "project/train_ddp.py found", 10, 10 if entry.exists() else 0)
    if not data:
        # no shared stream configured: score the submitted numbers instead of re-running
        t = json.loads((project / "outputs" / "throughput.json").read_text())
        rows.append(t)
        test("throughput.json", "tokens_per_sec" in t and "gpus" in t, f"{t.get('tokens_per_sec', 0):,.0f} tokens/s on {t.get('gpus')} GPU(s)", 30, 30)
    else:
        available = int(os.environ.get("ATELIER_GPUS", "1") or 1)
        for gpus in ([0] if cpu else [g for g in (1, 4) if g <= available]):
            env = dict(os.environ, ATELIER_GPUS=str(gpus))
            progress(15 + 40 * (gpus == 4), "running the student recipe on CPU, one process" if cpu else f"running the student recipe on {gpus} GPU(s)")
            proc = subprocess.run(["python", str(entry), "--data", data, "--iters", "200"], cwd=str(project), env=env, capture_output=True, text=True, timeout=4 * 2400 if cpu else 2400)
            if proc.returncode != 0:
                raise RuntimeError("\n".join((proc.stdout + proc.stderr).splitlines()[-15:]))
            rows.append(json.loads((project / "outputs" / "throughput.json").read_text()))
        test("runs on CPU" if cpu else "runs at both widths", len(rows) >= 1, "completed on CPU, one process (the 4-GPU run needs a GPU node)" if cpu else f"completed {len(rows)} run(s)", 20, 20)

    best = max(rows, key=lambda r: r["tokens_per_sec"])
    per_gpu = best["tokens_per_sec"] / max(best.get("gpus", 1), 1)
    # 40 points on throughput per GPU: nothing below 20k tokens/s, full marks at 200k
    import math

    pts = 40 * min(1.0, max(0.0, (math.log10(max(per_gpu, 1)) - 4.3) / 1.0))
    if possible < 100:
        test("throughput", True, f"{best['tokens_per_sec']:,.0f} tokens/s on CPU · not scored, the bar is set for GPUs", 0, 0)
    else:
        test("throughput", per_gpu > 20_000, f"{per_gpu:,.0f} tokens/s per GPU", 40, pts)
    R.metric("tokens_per_sec", "Throughput", best["tokens_per_sec"], "num", "kept", help="measured on CPU, one process" if cpu and data else None).metric("per_gpu", "Per GPU", per_gpu, "num", "sky", help="measured on CPU, one process" if cpu and data else None)
    vl = best.get("val_loss")
    if vl is not None:
        test("model quality", vl < 4.0, f"validation loss {vl:.3f}", 30, 30 * min(1.0, max(0.0, (5.0 - vl) / 2.5)))
        R.metric("val_loss", "Validation loss", vl, "num", "hold")
    if len(rows) > 1:
        one = min(rows, key=lambda r: r["gpus"])
        eff = (best["tokens_per_sec"] / one["tokens_per_sec"]) / (best["gpus"] / one["gpus"])
        R.metric("efficiency", "Scaling efficiency", eff, "pct", "raw")
        R.chart("scaling", "Throughput by GPU count", [{"gpus": r["gpus"], "tokens_per_sec": r["tokens_per_sec"]} for r in sorted(rows, key=lambda r: r["gpus"])], "gpus", [{"key": "tokens_per_sec", "label": "Tokens/s", "color": "kept"}], "line")
except Exception as exc:
    test("run", False, f"{type(exc).__name__}: {str(exc)[:200]}", 0, 0)
    traceback.print_exc()

score = round(min(100, score * 100 / possible), 1)
if possible < 100:
    R.note("Graded on a CPU-only node: the recipe ran once, as one process on the CPU, so throughput and scaling were not scored and the remaining checks are scaled to 100. Regrade on a GPU node for the full score.")
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
