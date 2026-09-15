"""Step 1 — latency, throughput, memory, and the quadratic curve; the model and questions the next steps use."""
import json
import os
import time
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, write_jsonl
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import load_tokenizer
from atelier_world import World
from atelier_world.prepared import choose_model, model_outputs, read_qa, train_val

parse_args()
P = params({"model_source": "generated", "data_source": "generated", "lengths": ["32", "64", "128", "256"], "batch_sizes": ["1", "4", "16", "64"], "n_prompts": 32})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("model_run_run")
run_model_key = "sft_model" if ref and ref["outputs"].get("sft_model") else "model"
# every later step works on the Atelier model itself — its uncached generation, its
# layers to quantise, its cache to speculate with, its logits to distil — so HuggingFace is refused here
info = choose_model(P, I, run_key="model_run", run_model_key=run_model_key, hint="Choose a model: a Fine-tuning step 3 run, a Pretraining step 3 run, or a prepared Atelier model.", formats=("atelier",))
model_path = info["model"]
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=info["lang"], seed=88)
model, ck = MiniLM.load(model_path, device)
info["tokenizer"] = info["tokenizer"] or ck.get("tokenizer")
tok = load_tokenizer(info["tokenizer"])
system = info["system"] or ck.get("system") or world.system_prompt
info["system"] = system
prepared = P["data_source"] == "prepared"
if prepared:
    # the prepared dataset's questions stand in for the generated ones in every step;
    # its training rows are what step 4 distils on
    train_qa, val_qa = train_val(P["data_material"], read_qa, 500, seed=88, limit=60000)

    def target(q):
        body = ("\n".join(q["steps"]) + "\n") if q["steps"] else ""
        return f"{body}{world.answer_prefix} {q['answer']}"

    write_jsonl(run_dir / "train.jsonl", [{"id": i, "family": q["family"], "difficulty": q["difficulty"], "prompt": q["prompt"], "target": target(q), "answer": q["answer"], "steps": q["steps"]} for i, q in enumerate(train_qa)])
    write_jsonl(run_dir / "val.jsonl", [{"id": i, "family": q["family"], "difficulty": q["difficulty"], "prompt": q["prompt"], "answer": q["answer"], "steps": q["steps"]} for i, q in enumerate(val_qa)])
    tasks = val_qa[: int(P["n_prompts"])]
    data_label = f"materials/{P['data_material']}"
else:
    tasks = world.eval_set(int(P["n_prompts"]), seed=333_999)
    data_label = "generated from the world"
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
R.note(f"Model: {info['label']}. Questions: {data_label}.")
R.output("baseline", str(run_dir / "baseline.json")).output("tokens_per_sec", best["tokens_per_sec"]).output("weight_bytes", weight_bytes)
for k, v in model_outputs(info, "model").items():
    R.output(k, v)
R.output("data_source", P["data_source"]).output("data_label", data_label)
if prepared:
    R.output("data_train", str(run_dir / "train.jsonl")).output("data_val", str(run_dir / "val.jsonl"))
R.save()
