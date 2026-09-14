"""Step 2 — size the model and estimate its training cost (CPU only)."""
import json
import os
import sys
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.model import GPTConfig, estimate  # noqa: E402

parse_args()
P = params({"n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256, "dropout": 0.0, "bias": False, "batch_tokens": 65536})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
vocab = int(I.get("vocab_size") or 50257)
vocab_padded = ((vocab + 63) // 64) * 64
if int(P["n_embd"]) % int(P["n_head"]) != 0:
    raise SystemExit(f"Embedding width {P['n_embd']} must be divisible by the number of heads {P['n_head']}.")
cfg = GPTConfig(vocab_size=vocab_padded, block_size=int(P["block_size"]), n_layer=int(P["n_layer"]), n_head=int(P["n_head"]), n_embd=int(P["n_embd"]), dropout=float(P["dropout"]), bias=bool(P["bias"]))
est = estimate(cfg)
(run_dir / "model_config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

train_tokens = int(I.get("train_tokens") or 0)
batch_tokens = int(P["batch_tokens"])
# memory: weights + grads + 2 AdamW moments in fp32 → 16 bytes/param, activations ≈ 34·T·d·L per sequence (bf16 with SDPA, rough)
weights_gb = est["total"] * 16 / 1e9
act_gb = batch_tokens * cfg.n_embd * cfg.n_layer * 34 / 1e9 + batch_tokens * cfg.vocab_size * 4 / 1e9
a6000_tflops = 155 * 0.35  # bf16 dense peak × a typical MFU for small models
flops_per_step = est["flops_per_token"] * batch_tokens
step_sec = flops_per_step / (a6000_tflops * 1e12)

R = Result()
R.metric("params", "Parameters", est["total"], "int", "kept", help=f"{est['non_embedding']:,} excluding embeddings")
R.metric("flops_per_token", "Training FLOPs per token", est["flops_per_token"], "int", "sky", help="≈ 6·N plus attention")
R.metric("chinchilla", "Compute-optimal tokens (20·N)", est["chinchilla_tokens"], "int", "hold", help=f"you have {train_tokens:,} training tokens" if train_tokens else None)
R.metric("mem_gb", "Memory estimate per GPU", weights_gb + act_gb, "num", "raw", help=f"{weights_gb:.2f} GB weights/optimizer + {act_gb:.2f} GB activations at {batch_tokens:,} tokens per step; A6000 has 48 GB")
R.metric("step_sec", "Estimated seconds per step", step_sec, "num", "raw", help="one A6000 at ~35% MFU")
R.chart("parts", "Parameters by component", [{"part": k, "params": est[k]} for k in ("embedding", "position", "attention", "mlp", "layernorm")], "part", [{"key": "params", "label": "Parameters", "color": "kept"}], "bar")
budget = []
for tok_mult in (1, 5, 10, 20, 40):
    tokens = est["non_embedding"] * tok_mult
    budget.append({"tokens_per_param": f"{tok_mult}×", "tokens": tokens, "hours": tokens * est["flops_per_token"] / (a6000_tflops * 1e12) / 3600})
R.chart("budget", "Training time on one A6000 by token budget", budget, "tokens_per_param", [{"key": "hours", "label": "Hours", "color": "raw"}], "bar", note="Multiples of the non-embedding parameter count.")
R.table("layers", "Shapes per block", [{"key": "name", "label": "Tensor"}, {"key": "shape", "label": "Shape"}, {"key": "params", "label": "Parameters", "fmt": "int"}], [
    {"name": "wte", "shape": f"{cfg.vocab_size}×{cfg.n_embd}", "params": cfg.vocab_size * cfg.n_embd},
    {"name": "wpe", "shape": f"{cfg.block_size}×{cfg.n_embd}", "params": cfg.block_size * cfg.n_embd},
    {"name": "attn.qkv (×L)", "shape": f"{cfg.n_embd}×{3 * cfg.n_embd}", "params": 3 * cfg.n_embd * cfg.n_embd},
    {"name": "attn.proj (×L)", "shape": f"{cfg.n_embd}×{cfg.n_embd}", "params": cfg.n_embd * cfg.n_embd},
    {"name": "mlp.fc (×L)", "shape": f"{cfg.n_embd}×{4 * cfg.n_embd}", "params": 4 * cfg.n_embd * cfg.n_embd},
    {"name": "mlp.proj (×L)", "shape": f"{4 * cfg.n_embd}×{cfg.n_embd}", "params": 4 * cfg.n_embd * cfg.n_embd},
])
R.artifact(run_dir / "model_config.json", "model_config.json")
R.output("model_config", str(run_dir / "model_config.json")).output("params", est["total"])
for k in ("train_bin", "val_bin", "meta", "vocab_size", "train_tokens"):
    if k in I:
        R.output(k, I[k])
R.save()
