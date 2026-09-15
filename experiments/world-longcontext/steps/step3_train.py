"""Step 3 — a short continuation at the stretched scale."""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import pretrain
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_needle import make_haystack, plant  # noqa: E402

parse_args()
P = params({"max_iters": 400, "batch_size": 4, "lr": 5e-5, "docs": 3000, "needle_share": 0.2})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=77)
tok = MiniTokenizer.load(I["tokenizer"])
target = int(I["target_length"])
rng = random.Random(31)
# prepared filler from step 1 (the documents set aside for training), if any
passages = json.loads(Path(I["filler_train"]).read_text(encoding="utf-8")) if I.get("data_source") == "prepared" and I.get("filler_train") else None

texts = []
n = int(P["docs"])
for i in range(n):
    if rng.random() < float(P["needle_share"]):
        nd = plant(world, tok, target - 60, rng.random(), rng, passages)
        texts.append(f"{nd['document']}\n\n{nd['question']}\n{world.answer_prefix} {nd['answer']}")
    else:
        texts.append(make_haystack(world, tok, target, rng, passages)[0])
    if i % 300 == 0:
        progress(3 + 25 * i / n, f"generated {i:,} long documents")
stats = pack(texts, tok, run_dir / "long.bin", max_tokens=target * n)
arr = np.fromfile(run_dir / "long.bin", dtype=np.uint16 if stats["dtype"] == "uint16" else np.uint32)
n_val = max(target * 8, arr.size // 20)
arr[:-n_val].tofile(run_dir / "train.bin")
arr[-n_val:].tofile(run_dir / "val.bin")
(run_dir / "long.bin").unlink()
progress(32, f"{arr.size:,} tokens packed")

model, ck = MiniLM.load(I["model"], device)
model.config.block_size = target
model.config.rope_scale = float(I["rope_scale"])
model.config.rope_base = float(I["rope_base"])
model._cos = model._sin = None
iters = int(P["max_iters"])
res = pretrain(model, TokenStream(run_dir / "train.bin", stats["dtype"]), TokenStream(run_dir / "val.bin", stats["dtype"]), run_dir,
               max_iters=iters, batch_size=int(P["batch_size"]), block_size=target, lr=float(P["lr"]), warmup=max(10, iters // 20), eval_every=max(25, iters // 10), device=device,
               on_log=lambda r: progress(35 + 60 * r["step"] / iters, f"step {r['step']}/{iters}" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else ""), step=r["step"], **{k: v for k, v in r.items() if k in ("val_loss", "loss")}))
model.save(run_dir / "model.pt", {"long": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": I.get("system"), "base_model": I["model"], "rope_scale": float(I["rope_scale"]), "rope_base": float(I["rope_base"]), "context": target})

R = Result()
R.metric("val_loss", f"Validation loss at {target} tokens", res["best_val_loss"], "num", "kept")
R.metric("context", "Context length", target, "int", "sky", help=f"was {I['trained_length']}")
R.metric("tokens_seen", "Tokens seen", res["tokens_seen"], "int", "raw")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "hold")
R.chart("loss", "Loss during the continuation", [{"step": h["step"], "train": h["train_loss"], "val": h["val_loss"]} for h in res["history"]], "step", [{"key": "train", "label": "Train", "color": "kept"}, {"key": "val", "label": "Validation", "color": "hold"}], "line", y_log=True)
R.artifact(run_dir / "model.pt", "model.pt (long context)")
R.output("long_model", str(run_dir / "model.pt")).output("val_loss", res["best_val_loss"])
for k in ("model", "tokenizer", "lang", "system", "target_length", "rope_scale", "rope_base", "trained_length", "model_format", "model_label", "data_source", "data_label", "filler_eval"):
    if k in I:
        R.output(k, I[k])
R.save()
