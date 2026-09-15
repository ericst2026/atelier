"""Step 2 — choose the shape and count what it will cost (CPU only)."""
import json
import os
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args
from atelier_mini.model import PRESETS, MiniConfig, estimate

parse_args()
P = params({"preset": "small", "n_layer": 8, "n_head": 8, "n_embd": 512, "block_size": 512, "dropout": 0.0, "tokens_per_step": 131072})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
vocab = int(I.get("vocab_size") or 4096)
train_tokens = int(I.get("train_tokens") or 0)

if P["preset"] == "custom":
    cfg = MiniConfig(vocab_size=vocab, n_layer=int(P["n_layer"]), n_head=int(P["n_head"]), n_embd=int(P["n_embd"]), block_size=int(P["block_size"]), dropout=float(P["dropout"]))
else:
    cfg = MiniConfig.preset(P["preset"], vocab, dropout=float(P["dropout"]))
    cfg.block_size = int(P["block_size"])
if cfg.n_embd % cfg.n_head:
    raise SystemExit(f"Width {cfg.n_embd} must divide by {cfg.n_head} heads.")
est = estimate(cfg)
(run_dir / "model_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

A6000_BF16_TFLOPS = 155 * 0.35   # dense peak, times a realistic utilisation for small models
step_tokens = int(P["tokens_per_step"])
step_sec = est["flops_per_token"] * step_tokens / (A6000_BF16_TFLOPS * 1e12)
mem_gb = est["total"] * 16 / 1e9 + step_tokens * cfg.n_embd * cfg.n_layer * 24 / 1e9 + step_tokens * cfg.vocab_size * 2 / 1e9

rows = []
for name, (L, H, D, T) in PRESETS.items():
    e = estimate(MiniConfig(vocab_size=vocab, n_layer=L, n_head=H, n_embd=D, block_size=T))
    hours = e["chinchilla_tokens"] * e["flops_per_token"] / (A6000_BF16_TFLOPS * 1e12) / 3600
    rows.append({"preset": name, "params": e["total"], "tokens": e["chinchilla_tokens"], "hours": hours, "shape": f"{L}L × {H}H × {D}d × {T}ctx"})

R = Result()
R.metric("params", "Parameters", est["total"], "int", "kept", help=f"{est['non_embedding']:,} outside the embedding table")
R.metric("embedding_share", "Embedding share", est["embedding"] / est["total"], "pct", "raw", help="tied input and output embeddings; a smaller vocabulary shifts parameters into the layers")
R.metric("chinchilla", "Compute-optimal tokens", est["chinchilla_tokens"], "int", "hold", help=f"you packed {train_tokens:,}" if train_tokens else "about 20 per parameter")
R.metric("step_sec", "Seconds per step", step_sec, "num", "sky", help=f"one A6000 at {step_tokens:,} tokens per step")
R.metric("mem_gb", "Memory estimate", mem_gb, "num", "raw", help="weights, gradients, AdamW moments and activations; an A6000 has 48 GB")
R.chart("parts", "Parameters by component", [{"part": k, "params": est[k]} for k in ("embedding", "attention", "mlp", "norms")], "part", [{"key": "params", "label": "Parameters", "color": "kept"}], "bar")
R.chart("presets", "What each size costs to train well", rows, "preset", [{"key": "params", "label": "Parameters", "color": "kept"}, {"key": "hours", "label": "Hours on one A6000", "color": "raw", "axis": "right"}], "bar", note="Hours assume the compute-optimal token budget. Fewer tokens is a valid choice — the loss is simply higher.")
R.table("shapes", "Presets", [{"key": "preset", "label": "Preset"}, {"key": "shape", "label": "Shape"}, {"key": "params", "label": "Parameters", "fmt": "int"}, {"key": "tokens", "label": "Tokens (20/param)", "fmt": "int"}, {"key": "hours", "label": "Hours", "fmt": "num"}], rows)
if train_tokens:
    ratio = train_tokens / max(1, est["total"])
    R.note(f"Your stream holds {ratio:.1f} tokens per parameter. Below about 10 the model is starved and the loss curve is still falling when the steps run out; far above 20 you are repeating data the model has already fitted.")
R.artifact(run_dir / "model_config.json", "model_config.json")
R.output("model_config", str(run_dir / "model_config.json")).output("params", est["total"])
for k in ("train_bin", "val_bin", "meta", "vocab_size", "tokenizer", "lang", "train_tokens", "data_source", "data_label", "tokenizer_label"):
    if k in I:
        R.output(k, I[k])
R.save()
