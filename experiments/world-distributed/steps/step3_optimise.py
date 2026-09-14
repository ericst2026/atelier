"""Step 3 — gradient accumulation, activation checkpointing, precision."""
import json
import os
import sys
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.model import MiniConfig, estimate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_bench import mfu, run_torchrun  # noqa: E402

parse_args()
P = params({"gpus": 4, "grad_accum_values": ["1", "2", "4"], "try_checkpointing": True, "try_fp32": True, "iters": 25})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
profile = json.loads(Path(I["profile"]).read_text())
meta = json.loads(Path(I["meta"]).read_text())
cfg = MiniConfig(**profile["config"])
est = estimate(cfg)
worker = Path(__file__).resolve().parent / "ddp_worker.py"
available = int(os.environ.get("ATELIER_GPUS", "1") or 1)
gpus = min(int(P["gpus"]), max(available, 1))
batch = profile["best"]["batch"]

settings = [{"name": f"accumulation {g}", "grad_accum": int(g), "checkpointing": 0, "precision": "bf16", "batch": batch} for g in sorted(int(x) for x in P["grad_accum_values"])]
if bool(P["try_checkpointing"]):
    settings.append({"name": "checkpointing", "grad_accum": 1, "checkpointing": 1, "precision": "bf16", "batch": batch})
    settings.append({"name": "checkpointing, double batch", "grad_accum": 1, "checkpointing": 1, "precision": "bf16", "batch": batch * 2})
if bool(P["try_fp32"]):
    settings.append({"name": "float32", "grad_accum": 1, "checkpointing": 0, "precision": "fp32", "batch": max(1, batch // 2)})

rows = []
for i, s in enumerate(settings):
    out_json = run_dir / f"opt_{i}.json"
    try:
        r = run_torchrun(worker, gpus, ["--train-bin", I["train_bin"], "--dtype-name", meta["dtype"], "--config", json.dumps(cfg.to_dict()), "--batch", str(s["batch"]), "--grad-accum", str(s["grad_accum"]), "--iters", str(int(P["iters"])), "--precision", s["precision"], "--checkpointing", str(s["checkpointing"]), "--out", str(out_json)], out_json)
        r["name"] = s["name"]
        r["mfu"] = mfu(r["tokens_per_sec"], est["flops_per_token"], gpus)
        r["tokens_per_step"] = s["batch"] * cfg.block_size * s["grad_accum"] * gpus
        rows.append(r)
        progress(5 + 90 * (i + 1) / len(settings), f"{s['name']}: {r['tokens_per_sec']:,.0f} tokens/s, {r['peak_gb']:.1f} GB")
    except Exception as exc:
        rows.append({"name": s["name"], "tokens_per_sec": 0, "peak_gb": 0, "mfu": 0, "error": str(exc)[:120], "tokens_per_step": 0, "optimizer_share": 0})
        progress(5 + 90 * (i + 1) / len(settings), f"{s['name']}: failed")

ok = [r for r in rows if r["tokens_per_sec"] > 0]
best = max(ok, key=lambda r: r["tokens_per_sec"]) if ok else rows[0]
baseline = next((r for r in ok if r["name"] == "accumulation 1"), best)
for r in rows:
    r["relative"] = r["tokens_per_sec"] / max(baseline["tokens_per_sec"], 1e-9)
(run_dir / "optimise.json").write_text(json.dumps({"rows": rows, "best": best, "gpus": gpus}, indent=2))

R = Result()
R.metric("best_throughput", "Best throughput", best["tokens_per_sec"], "num", "kept", help=f"{best['name']} on {gpus} GPUs")
R.metric("gain", "Against the plain recipe", best["relative"], "num", "sky", help="1.0 means nothing was gained")
R.metric("best_memory", "Peak memory of the best setting", best["peak_gb"], "num", "hold")
R.metric("mfu", "Utilisation", best.get("mfu", 0), "pct", "raw")
R.chart("throughput", "Throughput by setting", rows, "name", [{"key": "tokens_per_sec", "label": "Tokens per second", "color": "kept"}], "bar")
R.chart("memory", "Memory against throughput", rows, "name", [{"key": "peak_gb", "label": "Peak GB", "color": "raw"}, {"key": "relative", "label": "Relative throughput", "color": "sky", "axis": "right"}], "bar", note="Checkpointing buys memory with compute: the bar on the left falls, and if you spend the saving on a larger batch the bar on the right can still rise.")
R.chart("sync", "Share of the step spent synchronising", rows, "name", [{"key": "optimizer_share", "label": "Sync share", "color": "dup"}], "bar", note="Accumulating gradients all-reduces less often, so this falls — at the cost of a larger effective batch, which is a change to the recipe, not a free win.")
R.table("rows", "Measurements", [{"key": "name", "label": "Setting"}, {"key": "tokens_per_sec", "label": "Tokens/s", "fmt": "int"}, {"key": "relative", "label": "Relative", "fmt": "num"}, {"key": "peak_gb", "label": "Peak GB", "fmt": "num"}, {"key": "tokens_per_step", "label": "Tokens/step", "fmt": "int"}, {"key": "error", "label": "Error"}], rows)
R.artifact(run_dir / "optimise.json", "optimise.json")
R.output("optimise", str(run_dir / "optimise.json")).output("best_grad_accum", best.get("grad_accum", 1)).output("gpus_used", gpus)
for k in ("profile", "scale", "train_bin", "val_bin", "meta", "vocab_size"):
    if k in I:
        R.output(k, I[k])
R.save()
