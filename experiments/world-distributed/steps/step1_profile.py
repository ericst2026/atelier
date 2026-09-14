"""Step 1 — one GPU, measured properly."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM, estimate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_bench import mfu, on_cpu, time_steps  # noqa: E402

parse_args()
P = params({"preset": "small", "block_size": 512, "batch_sizes": ["8", "16", "32", "64"], "iters": 30})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("pack_run_run")
if not ref:
    raise SystemExit("Choose a Pretraining step 1 run for the token stream.")
o = ref["outputs"]
meta = json.loads(Path(o["meta"]).read_text())
cpu = on_cpu()
device = "cpu" if cpu else "cuda"
cfg = MiniConfig.preset(P["preset"], int(o["vocab_size"]))
cfg.block_size = int(P["block_size"])
est = estimate(cfg)
stream = TokenStream(o["train_bin"], meta["dtype"])

rows = []
sizes = sorted(int(b) for b in P["batch_sizes"])
for i, batch in enumerate(sizes):
    torch.manual_seed(1)
    model = MiniLM(cfg).to(device)
    try:
        r = time_steps(model, stream, batch, cfg.block_size, device, int(P["iters"]))
        rows.append({"batch": batch, "tokens_per_sec": r["tokens_per_sec"], "sec_per_step": r["sec_per_step"], "peak_gb": r["peak_gb"], "data_share": r["data_share"], "mfu": mfu(r["tokens_per_sec"], est["flops_per_token"])})
        progress(10 + 85 * (i + 1) / len(sizes), f"batch {batch}: {r['tokens_per_sec']:,.0f} tokens/s, {r['peak_gb']:.1f} GB", step=batch, tokens_per_sec=r["tokens_per_sec"])
    except torch.cuda.OutOfMemoryError:
        rows.append({"batch": batch, "tokens_per_sec": 0, "sec_per_step": 0, "peak_gb": 0, "data_share": 0, "mfu": 0, "oom": True})
        progress(10 + 85 * (i + 1) / len(sizes), f"batch {batch}: out of memory")
    del model
    if not cpu:
        torch.cuda.empty_cache()

best = max(rows, key=lambda r: r["tokens_per_sec"])
(run_dir / "profile.json").write_text(json.dumps({"rows": rows, "best": best, "config": cfg.to_dict(), "flops_per_token": est["flops_per_token"], "params": est["total"]}, indent=2))

R = Result()
R.metric("tokens_per_sec", "Best throughput", best["tokens_per_sec"], "num", "kept", help=f"at batch {best['batch']}")
R.metric("mfu", "Model FLOPs utilisation", best["mfu"], "pct", "sky", help="not measured on CPU" if cpu else "share of the A6000's bf16 peak actually used")
R.metric("peak_gb", "Peak memory", best["peak_gb"], "num", "hold", help="not measured on CPU" if cpu else "of 48 GB")
R.metric("data_share", "Time spent fetching data", best["data_share"], "pct", "dup", help="anything large here means the GPU is waiting on the CPU")
R.chart("throughput", "Throughput against batch size", rows, "batch", [{"key": "tokens_per_sec", "label": "Tokens per second", "color": "kept"}, {"key": "peak_gb", "label": "Peak GB", "color": "raw", "axis": "right"}], "line", note="Measured on a CPU-only node with one process: no GPU, no bf16, and nothing to scale across, so the numbers are not comparable with a GPU run." if cpu else "Throughput flattens once the GPU is saturated; past that point a larger batch only costs memory.")
R.chart("mfu", "Utilisation against batch size", rows, "batch", [{"key": "mfu", "label": "MFU", "color": "sky"}], "line", y_domain=[0, 1], note="MFU is measured against a GPU's peak, so it is reported as 0 on CPU." if cpu else None)
R.table("rows", "Measurements", [{"key": "batch", "label": "Batch"}, {"key": "tokens_per_sec", "label": "Tokens/s", "fmt": "int"}, {"key": "sec_per_step", "label": "Seconds/step", "fmt": "num"}, {"key": "peak_gb", "label": "Peak GB", "fmt": "num"}, {"key": "mfu", "label": "MFU", "fmt": "pct"}, {"key": "data_share", "label": "Data wait", "fmt": "pct"}], rows)
if cpu:
    R.note("Measured on a CPU-only node with one process: no GPU, no bf16, and nothing to scale across, so the numbers are not comparable with a GPU run.")
R.artifact(run_dir / "profile.json", "profile.json")
R.output("profile", str(run_dir / "profile.json")).output("best_batch", best["batch"]).output("single_gpu_tps", best["tokens_per_sec"])
R.output("train_bin", o["train_bin"]).output("val_bin", o["val_bin"]).output("meta", o["meta"]).output("vocab_size", o["vocab_size"]).output("preset", P["preset"]).output("block_size", cfg.block_size)
R.save()
