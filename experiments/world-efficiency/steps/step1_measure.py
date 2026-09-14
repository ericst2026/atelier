"""Step 1 — latency, throughput, memory, and the quadratic curve."""
import json
import os
import time
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"lengths": ["32", "64", "128", "256"], "batch_sizes": ["1", "4", "16", "64"], "n_prompts": 32})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("model_run_run")
if not ref:
    raise SystemExit("Choose a model: a Fine-tuning step 3 run, or a Pretraining step 3 run.")
o = ref["outputs"]
model_path = o.get("sft_model") or o.get("model")
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=o.get("lang", "en"), seed=88)
model, ck = MiniLM.load(model_path, device)
tok = MiniTokenizer.load(o["tokenizer"])
system = o.get("system") or ck.get("system") or world.system_prompt
tasks = world.eval_set(int(P["n_prompts"]), seed=333_999)
prompts = [t["prompt"] for t in tasks]

weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
lengths = sorted(int(x) for x in P["lengths"])
batches = sorted(int(x) for x in P["batch_sizes"])

length_rows = []
for i, L in enumerate(lengths):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    generate(model, tok, prompts[:8], L, 0.0, batch_size=8, system=system)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    length_rows.append({"length": L, "seconds": dt, "per_token_ms": dt / (L * 8) * 1000, "tokens_per_sec": L * 8 / dt})
    progress(5 + 45 * (i + 1) / len(lengths), f"{L} tokens: {dt:.2f}s")

batch_rows = []
for i, B in enumerate(batches):
    n = max(B, 8)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    generate(model, tok, (prompts * 10)[:n], 64, 0.0, batch_size=B, system=system)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    batch_rows.append({"batch": B, "tokens_per_sec": n * 64 / dt, "latency_ms": dt / (n / B) * 1000, "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0})
    progress(50 + 45 * (i + 1) / len(batches), f"batch {B}: {batch_rows[-1]['tokens_per_sec']:,.0f} tokens/s")

best = max(batch_rows, key=lambda r: r["tokens_per_sec"])
(run_dir / "baseline.json").write_text(json.dumps({"lengths": length_rows, "batches": batch_rows, "weight_bytes": weight_bytes}, indent=2))

R = Result()
R.metric("tokens_per_sec", "Best throughput", best["tokens_per_sec"], "num", "kept", help=f"at batch {best['batch']}")
R.metric("latency", "Latency for one request", batch_rows[0]["latency_ms"], "num", "sky", help="batch 1 — what a single user waits")
R.metric("weights_mb", "Weights in memory", weight_bytes / 1e6, "num", "hold")
R.metric("per_token", "Milliseconds per token at 256", length_rows[-1]["per_token_ms"], "num", "raw", help=f"at 32 tokens: {length_rows[0]['per_token_ms']:.2f} ms — the gap is the missing cache")
R.chart("quadratic", "Cost per token as the output grows", length_rows, "length", [{"key": "per_token_ms", "label": "Milliseconds per token", "color": "dup"}], "line", note="Rising, because without a cache every step re-reads the whole prefix. The next steps flatten this line.")
R.chart("batching", "Throughput and latency against batch size", batch_rows, "batch", [{"key": "tokens_per_sec", "label": "Tokens per second", "color": "kept"}, {"key": "latency_ms", "label": "Latency, ms", "color": "raw", "axis": "right"}], "line", note="Batching buys throughput and costs latency. Which one you are optimising is a product decision, not a technical one.")
R.table("rows", "Measurements", [{"key": "batch", "label": "Batch"}, {"key": "tokens_per_sec", "label": "Tokens/s", "fmt": "int"}, {"key": "latency_ms", "label": "Latency ms", "fmt": "num"}, {"key": "peak_gb", "label": "Peak GB", "fmt": "num"}], batch_rows, note=None if device == "cuda" else "This run was on CPU: peak GB is GPU memory and was not measured, so it reads 0. Throughput and latency are CPU numbers.")
R.artifact(run_dir / "baseline.json", "baseline.json")
R.output("baseline", str(run_dir / "baseline.json")).output("model", model_path).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("system", system).output("tokens_per_sec", best["tokens_per_sec"]).output("weight_bytes", weight_bytes)
R.save()
