"""Step 2 — the same job at one, two, four and eight GPUs."""
import json
import os
import sys
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.model import MiniConfig, estimate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_bench import mfu, on_cpu, run_torchrun  # noqa: E402

parse_args()
P = params({"widths": ["1", "2", "4", "8"], "batch_per_gpu": 32, "iters": 40, "keep_global_batch": False})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
profile = json.loads(Path(I["profile"]).read_text())
meta = json.loads(Path(I["meta"]).read_text())
cfg = MiniConfig(**profile["config"])
est = estimate(cfg)
worker = Path(__file__).resolve().parent / "ddp_worker.py"
available = int(os.environ.get("ATELIER_GPUS", "1") or 1)
cpu = on_cpu()
# on CPU: one gloo process, never several pretending to be GPUs
widths = [1] if cpu else [w for w in sorted(int(x) for x in P["widths"]) if w <= max(available, 1)]
if not widths:
    raise SystemExit(f"This run was given {available} GPU(s); ask for more on the step, or pick smaller widths.")

rows = []
base_batch = int(P["batch_per_gpu"])
for i, w in enumerate(widths):
    batch = max(1, base_batch * widths[0] // w) if bool(P["keep_global_batch"]) else base_batch
    out_json = run_dir / f"scale_{w}.json"
    r = run_torchrun(worker, w, ["--train-bin", I["train_bin"], "--dtype-name", meta["dtype"], "--config", json.dumps(cfg.to_dict()), "--batch", str(batch), "--iters", str(int(P["iters"])), "--out", str(out_json)], out_json)
    r["mfu"] = mfu(r["tokens_per_sec"], est["flops_per_token"], w)
    rows.append(r)
    progress(10 + 85 * (i + 1) / len(widths), f"{w} GPU(s): {r['tokens_per_sec']:,.0f} tokens/s", step=w, tokens_per_sec=r["tokens_per_sec"])

one = next((r for r in rows if r["gpus"] == 1), rows[0])
for r in rows:
    r["speedup"] = r["tokens_per_sec"] / one["tokens_per_sec"]
    r["efficiency"] = r["speedup"] / (r["gpus"] / one["gpus"])
    r["ideal"] = r["gpus"] / one["gpus"]
(run_dir / "scale.json").write_text(json.dumps(rows, indent=2))
top = max(rows, key=lambda r: r["gpus"])

R = Result()
R.metric("best_throughput", "Throughput at the widest", top["tokens_per_sec"], "num", "kept", help="one process on CPU" if cpu else f"{top['gpus']} GPUs")
R.metric(f"efficiency_{top['gpus']}", f"Efficiency at {top['gpus']} GPUs", top["efficiency"], "pct", "sky", help="one process on CPU, so there is no scaling to measure" if cpu else "speedup divided by the number of GPUs; 100% would be perfect")
R.metric("optimizer_share", "Time in the all-reduce and optimizer", top["optimizer_share"], "pct", "dup", help="what the synchronisation costs")
R.metric("mfu", "Utilisation per GPU", top["mfu"], "pct", "hold", help="not measured on CPU" if cpu else None)
R.chart("speedup", "Speedup against GPU count", rows, "gpus", [{"key": "speedup", "label": "Measured", "color": "kept"}, {"key": "ideal", "label": "Perfect scaling", "color": "hold"}], "line", note="Measured on a CPU-only node with one process: no GPU, no bf16, and nothing to scale across, so the numbers are not comparable with a GPU run. The flat line is not a scaling result." if cpu else "The gap between the lines is the all-reduce. A small model has a short step, so the fixed communication cost is a larger share of it.")
R.chart("efficiency", "Efficiency", rows, "gpus", [{"key": "efficiency", "label": "Efficiency", "color": "sky"}], "line", y_domain=[0, 1], note="One CPU process: efficiency is 100% by definition." if cpu else None)
R.chart("memory", "Peak memory per GPU", rows, "gpus", [{"key": "peak_gb", "label": "GB", "color": "raw"}], "bar", note="GPU memory is not measured on CPU." if cpu else "Data parallelism replicates the model on every GPU, so memory per GPU does not fall as the node widens.")
R.table("rows", "Measurements", [{"key": "gpus", "label": "GPUs"}, {"key": "batch_per_gpu", "label": "Batch/GPU"}, {"key": "tokens_per_sec", "label": "Tokens/s", "fmt": "int"}, {"key": "speedup", "label": "Speedup", "fmt": "num"}, {"key": "efficiency", "label": "Efficiency", "fmt": "pct"}, {"key": "optimizer_share", "label": "Sync share", "fmt": "pct"}, {"key": "peak_gb", "label": "Peak GB", "fmt": "num"}], rows)
if cpu:
    R.note("Measured on a CPU-only node with one process: no GPU, no bf16, and nothing to scale across, so the numbers are not comparable with a GPU run. Run this step on a GPU node to see where the speedup stops.")
R.artifact(run_dir / "scale.json", "scale.json")
R.output("scale", str(run_dir / "scale.json")).output("profile", I["profile"]).output("train_bin", I["train_bin"]).output("val_bin", I["val_bin"]).output("meta", I["meta"]).output("vocab_size", I["vocab_size"])
R.save()
