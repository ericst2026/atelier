"""Step 4 — train on it, and on the two corpora that bracket it."""
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import pretrain
from atelier_world import World

parse_args()
P = params({"preset": "tiny", "max_iters": 800, "batch_size": 32, "lr": 8e-4, "compare_baselines": True, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("tokenizer_run_run")
if not ref:
    raise SystemExit("Choose a Tokenizer run (step 3) — the three corpora have to be measured with the same vocabulary.")
tok = MiniTokenizer.load(ref["outputs"]["tokenizer"])
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=int(I.get("seed", 17)))

everything = read_jsonl(I["corpus"])
yours = read_jsonl(I["clean"])
perfect = [d for d in everything if d["clean"]]
corpora = {"your filtered corpus": [d["text"] for d in yours]}
if bool(P["compare_baselines"]):
    corpora["everything, unfiltered"] = [d["text"] for d in everything]
    corpora["the clean documents (ceiling)"] = [d["text"] for d in perfect]

# held-out text the generator makes fresh, so no corpus has an advantage on it
val_texts = [d["text"] for d in world.documents(3000, seed=90_001)]
vs = pack(val_texts, tok, run_dir / "val.bin", max_tokens=1_500_000)
val = TokenStream(run_dir / "val.bin", vs["dtype"])
cfg = MiniConfig.preset(P["preset"], tok.vocab_size)
cfg.block_size = 256
iters = int(P["max_iters"])
results, curves = [], {}

for i, (name, texts) in enumerate(corpora.items()):
    progress(5 + 85 * i / len(corpora), f"packing and training: {name}")
    st = pack(texts, tok, run_dir / f"train_{i}.bin", max_tokens=200_000_000)
    torch.manual_seed(int(P["seed"]))
    model = MiniLM(cfg).to(device)
    res = pretrain(model, TokenStream(run_dir / f"train_{i}.bin", st["dtype"]), val, run_dir / f"m{i}",
                   max_iters=iters, batch_size=int(P["batch_size"]), block_size=cfg.block_size, lr=float(P["lr"]),
                   warmup=max(20, iters // 20), eval_every=max(25, iters // 10), device=device,
                   on_log=lambda r, i=i, name=name: progress(5 + 85 * (i + r["step"] / iters) / len(corpora), f"{name} · step {r['step']}/{iters}" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else ""), step=r["step"], **{k: v for k, v in r.items() if k == "val_loss"}))
    results.append({"corpus": name, "docs": st["docs"], "tokens": st["tokens"], "val_loss": res["best_val_loss"]})
    for h in res["history"]:
        curves.setdefault(h["step"], {"step": h["step"]})[name] = h["val_loss"]
    del model
    torch.cuda.empty_cache()

mine = results[0]
ceiling = next((r for r in results if "ceiling" in r["corpus"]), None)
raw = next((r for r in results if "unfiltered" in r["corpus"]), None)
best = min(results, key=lambda r: r["val_loss"])

R = Result()
R.metric("val_loss", "Your corpus", mine["val_loss"], "num", "kept", help=f"{mine['docs']:,} documents, {mine['tokens']:,} tokens")
if raw:
    R.metric("unfiltered", "Everything, unfiltered", raw["val_loss"], "num", "dup", help=f"{raw['tokens']:,} tokens")
if ceiling:
    R.metric("ceiling", "Perfect filtering", ceiling["val_loss"], "num", "hold", help="the clean documents only — no filter can do better")
    R.metric("vs_ceiling", "Distance to perfect", mine["val_loss"] - ceiling["val_loss"], "num", "sky")
R.metric("f1", "F1 from step 2", I.get("f1"), "num", "raw", help="how the same filter scored against the labels")
R.chart("losses", "Validation loss by corpus", [{"corpus": r["corpus"], "val_loss": r["val_loss"]} for r in results], "corpus", [{"key": "val_loss", "label": "Validation loss", "color": "kept"}], "bar", note="Identical model, identical steps, identical tokenizer. The only difference is which documents survived.")
R.chart("curves", "Loss during training", [curves[k] for k in sorted(curves)], "step", [{"key": r["corpus"], "label": r["corpus"]} for r in results], "line", y_log=True)
R.chart("size", "Tokens against loss", [{"corpus": r["corpus"], "tokens": r["tokens"], "val_loss": r["val_loss"]} for r in results], "corpus", [{"key": "tokens", "label": "Tokens", "color": "sky"}, {"key": "val_loss", "label": "Validation loss", "color": "kept", "axis": "right"}], "bar", note="This is where precision and recall stop being the whole story: a filter can be right about every document it removed and still leave too little text to train on.")
R.table("results", "Results", [{"key": "corpus", "label": "Corpus"}, {"key": "docs", "label": "Documents", "fmt": "int"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "val_loss", "label": "Validation loss", "fmt": "num"}], results)
if best["corpus"] != mine["corpus"]:
    R.note(f"Your filtered corpus was not the best of the three — {best['corpus']} was. Compare the token counts before changing a threshold: the usual cause is a filter that removed too much rather than the wrong things.")
R.output("val_loss", mine["val_loss"]).output("clean", I["clean"]).output("f1", I.get("f1"))
R.save()
