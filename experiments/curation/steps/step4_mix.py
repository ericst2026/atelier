"""Step 4 — train on the mix and on the baselines, identically."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf
from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import HFTokenizer
from atelier_mini.train import pretrain

parse_args()
P = params({"curated_share": 0.7, "reference_share": 0.3, "max_tokens": 20_000_000, "preset": "tiny", "max_iters": 1500, "compare_baselines": True, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = HFTokenizer(hf.model_path("gpt2") / "tokenizer.json")
rng = random.Random(int(P["seed"]))
budget = int(P["max_tokens"])

curated = [d["text"] for d in hf.read_jsonl(I["clean"]) if d.get("text")]
reference = [d["text"] for d in hf.dataset_split("curation", "fineweb-edu", "train", limit=60000) if d.get("text")]
raw = [d["text"] for d in hf.dataset_split("curation", "c4-raw", "train", limit=60000) if d.get("text")]
val_rows = [d["text"] for d in hf.dataset_split("pretrain", "wikitext-103", "validation", limit=4000) if d.get("text")]
progress(5, f"{len(curated):,} curated, {len(reference):,} reference, {len(raw):,} raw documents")

cs, rs = float(P["curated_share"]), float(P["reference_share"])
total = cs + rs or 1.0
cs, rs = cs / total, rs / total
mix = []
while len(mix) < max(len(curated), len(reference)) * 2 and (curated or reference):
    pool = curated if (rng.random() < cs and curated) else reference
    if not pool:
        pool = curated or reference
    mix.append(rng.choice(pool))
    if len(mix) > 200000:
        break

corpora = {"your mix": mix}
if bool(P["compare_baselines"]):
    corpora["raw only"] = raw
    corpora["FineWeb-Edu only"] = reference

val_stats = pack(val_rows, tok, run_dir / "val.bin", max_tokens=budget // 20)
val = TokenStream(run_dir / "val.bin", val_stats["dtype"])
cfg = MiniConfig.preset(P["preset"], tok.vocab_size)
cfg.block_size = 256
results, curves = [], {}
iters = int(P["max_iters"])
for i, (name, texts) in enumerate(corpora.items()):
    progress(10 + 80 * i / len(corpora), f"packing and training: {name}")
    st = pack(texts, tok, run_dir / f"train_{i}.bin", max_tokens=budget)
    stream = TokenStream(run_dir / f"train_{i}.bin", st["dtype"])
    torch.manual_seed(int(P["seed"]))
    model = MiniLM(cfg).to(device)
    res = pretrain(model, stream, val, run_dir / f"m{i}", max_iters=iters, batch_size=32, block_size=cfg.block_size, lr=8e-4, warmup=max(20, iters // 20), eval_every=max(25, iters // 10), device=device,
                   on_log=lambda r, name=name, i=i: progress(10 + 80 * (i + r["step"] / iters) / len(corpora), f"{name} · step {r['step']}/{iters}" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else ""), step=r["step"], **{k: v for k, v in r.items() if k == "val_loss"}))
    results.append({"corpus": name, "tokens": st["tokens"], "docs": st["docs"], "val_loss": res["best_val_loss"]})
    for h in res["history"]:
        curves.setdefault(h["step"], {"step": h["step"]})[name] = h["val_loss"]
    del model
    torch.cuda.empty_cache()

best = min(results, key=lambda r: r["val_loss"])
yours = next((r for r in results if r["corpus"] == "your mix"), best)
R = Result()
R.metric("val_loss", "Validation loss of your mix", yours["val_loss"], "num", "kept")
for r in results:
    if r["corpus"] != "your mix":
        R.metric(f"loss_{r['corpus'].split()[0].lower()}", f"Loss · {r['corpus']}", r["val_loss"], "num", "raw")
R.metric("tokens", "Tokens trained on", yours["tokens"], "int", "sky")
R.metric("rank", "Best corpus", best["corpus"], "text", "kept" if best["corpus"] == "your mix" else "dup")
R.chart("losses", "Validation loss by corpus", [{"corpus": r["corpus"], "val_loss": r["val_loss"]} for r in results], "corpus", [{"key": "val_loss", "label": "Validation loss", "color": "kept"}], "bar", note="Identical model, identical steps, identical tokenizer. The only difference is the data.")
R.chart("curves", "Loss during training", [curves[k] for k in sorted(curves)], "step", [{"key": r["corpus"], "label": r["corpus"]} for r in results], "line", y_log=True)
R.table("results", "Results", [{"key": "corpus", "label": "Corpus"}, {"key": "docs", "label": "Documents", "fmt": "int"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "val_loss", "label": "Validation loss", "fmt": "num"}], results)
if I.get("contamination", 0) > 0.01:
    R.note(f"About {float(I['contamination']):.1%} of your corpus shares long phrases with the validation set. That flatters every number on this page.")
R.output("val_loss", yours["val_loss"]).output("clean", I["clean"]).output("filter_params", I.get("filter_params"))
R.save()
