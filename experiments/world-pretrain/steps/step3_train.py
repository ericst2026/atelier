"""Step 3 — the training loop."""
import json
import math
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, curves
from atelier_mini.data import TokenStream
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.train import pretrain

parse_args()
P = params({"max_iters": 3000, "batch_size": 32, "grad_accum": 1, "lr": 8e-4, "warmup": 150, "weight_decay": 0.1, "eval_every": 100, "eval_iters": 20, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
meta = json.loads(Path(I["meta"]).read_text())
cfg = MiniConfig(**json.loads(Path(I["model_config"]).read_text()))
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(int(P["seed"]))

train = TokenStream(I["train_bin"], meta["dtype"])
val = TokenStream(I["val_bin"], meta["dtype"])
model = MiniLM(cfg).to(device)
max_iters = int(P["max_iters"])
tokens_per_step = int(P["batch_size"]) * cfg.block_size * int(P["grad_accum"])
print(f"[train] {model.num_params():,} parameters · {len(train):,} training tokens · {tokens_per_step:,} tokens/step · {device}", flush=True)


def on_log(row):
    if "val_loss" in row:
        progress(100 * row["step"] / max_iters, f"step {row['step']}/{max_iters} · val {row['val_loss']:.3f}", step=row["step"], val_loss=row["val_loss"], train_loss=row["train_loss"])
    else:
        tps = row.get("tokens_per_sec") or 0
        left = (max_iters - row["step"]) * int(P["batch_size"]) * int(P["grad_accum"]) * cfg.block_size / tps if tps else 0
        eta = f" · ~{left / 3600:.1f} h left" if left >= 5400 else f" · ~{left / 60:.0f} min left" if left else ""
        progress(100 * row["step"] / max_iters, f"step {row['step']}/{max_iters} · loss {row['loss']:.3f} · {tps:,.0f} tok/s{eta}", step=row["step"], **curves(row, "loss", "tokens_per_sec", "lr", "grad_norm"))


res = pretrain(model, train, val, run_dir, max_iters=max_iters, batch_size=int(P["batch_size"]), grad_accum=int(P["grad_accum"]), block_size=cfg.block_size, lr=float(P["lr"]), warmup=int(P["warmup"]), weight_decay=float(P["weight_decay"]), eval_every=int(P["eval_every"]), eval_iters=int(P["eval_iters"]), device=device, on_log=on_log)
hist = res["history"]

R = Result()
R.metric("val_loss", "Best validation loss", res["best_val_loss"], "num", "hold")
R.metric("perplexity", "Validation perplexity", math.exp(min(res["best_val_loss"], 20)), "num", "hold")
R.metric("params", "Parameters", model.num_params(), "int", "kept")
R.metric("tokens_seen", "Tokens seen", res["tokens_seen"], "int", "raw", help=f"{res['tokens_seen'] / max(1, model.num_params()):.1f} per parameter")
R.metric("tokens_per_sec", "Throughput", res["tokens_seen"] / max(res["elapsed_sec"], 1e-6), "num", "sky")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "sky")
R.chart("loss", "Loss", [{"step": h["step"], "train": h["train_loss"], "val": h["val_loss"]} for h in hist], "step", [{"key": "train", "label": "Train", "color": "kept"}, {"key": "val", "label": "Validation", "color": "hold"}], "line", y_log=True, note="If the two lines separate, the model is memorising the corpus — generate more documents or dedupe harder.")
R.chart("tokens", "Validation loss against tokens seen", [{"tokens": h["tokens"], "val": h["val_loss"]} for h in hist if h["step"]], "tokens", [{"key": "val", "label": "Validation loss", "color": "hold"}], "line", x_log=True, y_log=True)
R.chart("lr", "Learning rate", [{"step": h["step"], "lr": h["lr"]} for h in hist], "step", [{"key": "lr", "label": "lr", "color": "sky"}], "line")
R.table("history", "Evaluations", [{"key": "step", "label": "Step"}, {"key": "train_loss", "label": "Train", "fmt": "num"}, {"key": "val_loss", "label": "Validation", "fmt": "num"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "elapsed", "label": "Seconds", "fmt": "num"}], hist)
R.artifact(Path(res["checkpoint"]), "model.pt (best validation)")
R.output("model", res["checkpoint"]).output("val_loss", res["best_val_loss"]).output("params", model.num_params()).output("tokens_seen", res["tokens_seen"])
for k in ("train_bin", "val_bin", "meta", "tokenizer", "vocab_size", "lang", "model_config", "data_source", "data_label", "tokenizer_label"):
    if k in I:
        R.output(k, I[k])
R.save()
