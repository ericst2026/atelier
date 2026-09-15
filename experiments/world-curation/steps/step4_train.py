"""Step 4 — train on it, and on the two corpora that bracket it."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import load_tokenizer
from atelier_mini.train import pretrain
from atelier_world import World
from atelier_world.prepared import choose_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_signals import apply_rules  # noqa: E402

parse_args()
P = params({"tokenizer_source": "generated", "tokenizer_material": None, "preset": "tiny", "max_iters": 800, "batch_size": 32, "lr": 8e-4, "compare_baselines": True, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
# the vocabulary: a Tokenizer run, or the tokenizer.json of a prepared model (Atelier or HuggingFace)
tk = choose_model(P, I, run_key="tokenizer_run", run_model_key="tokenizer", source_key="tokenizer_source", material_key="tokenizer_material",
                  hint="Choose a Tokenizer run (step 3), or a prepared model whose tokenizer to use — the corpora have to be measured with the same vocabulary.")
tok_path = Path(tk["tokenizer"])
if tok_path.is_dir():
    tok_path = tok_path / "tokenizer.json"
if not tok_path.exists():
    raise SystemExit(f"{tk['label']} has no tokenizer.json to pack text with.")
tok = load_tokenizer(tok_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=int(I.get("seed", 17)))
labelled = I.get("data_source", "generated") != "prepared"

everything = read_jsonl(I["corpus"])
yours = read_jsonl(I["clean"])
corpora = {"your filtered corpus": [d["text"] for d in yours]}
if bool(P["compare_baselines"]):
    corpora["everything, unfiltered"] = [d["text"] for d in everything]
    if labelled:
        perfect = [d for d in everything if d["clean"]]
        corpora["the clean documents (ceiling)"] = [d["text"] for d in perfect]

if labelled:
    # held-out text the generator makes fresh, so no corpus has an advantage on it
    val_texts = [d["text"] for d in world.documents(3000, seed=90_001)]
    val_label = "3,000 fresh generated documents"
else:
    # a prepared corpus: the documents step 1 held out, kept only where your step-2
    # thresholds keep them — no labels say which are clean, so the filter stands in
    held = read_jsonl(I["heldout"])
    fp = json.loads(Path(I["filter_params"]).read_text()) if I.get("filter_params") else None
    val_texts = [d["text"] for d in held if fp is None or apply_rules(d["text"], fp)[0]]
    val_label = f"{len(val_texts):,} of the {len(held):,} held-out documents (those your filter keeps)"
    if not val_texts:
        val_texts = [d["text"] for d in held]
        val_label = f"all {len(held):,} held-out documents (your filter keeps none of them)"
vs = pack(val_texts, tok, run_dir / "val.bin", max_tokens=1_500_000)
if vs["tokens"] < 600:
    raise SystemExit(f"The validation text packs to only {vs['tokens']} tokens; step 4 needs a few hundred more. Use a larger corpus, or a larger held-out share in step 1.")
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
    (run_dir / f"m{i}").mkdir(parents=True, exist_ok=True)
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
if labelled:
    R.metric("f1", "F1 from step 2", I.get("f1"), "num", "raw", help="how the same filter scored against the labels")
else:
    R.note(f"{I.get('data_label', 'A prepared corpus')} has no labels: there is no perfectly-clean ceiling and no F1 to set beside these losses. Validation is on {val_label}, so a filter is partly judged by its own standard — compare against the unfiltered run rather than reading the number alone.")
R.chart("losses", "Validation loss by corpus", [{"corpus": r["corpus"], "val_loss": r["val_loss"]} for r in results], "corpus", [{"key": "val_loss", "label": "Validation loss", "color": "kept"}], "bar", note="Identical model, identical steps, identical tokenizer. The only difference is which documents survived.")
R.chart("curves", "Loss during training", [curves[k] for k in sorted(curves)], "step", [{"key": r["corpus"], "label": r["corpus"]} for r in results], "line", y_log=True)
R.chart("size", "Tokens against loss", [{"corpus": r["corpus"], "tokens": r["tokens"], "val_loss": r["val_loss"]} for r in results], "corpus", [{"key": "tokens", "label": "Tokens", "color": "sky"}, {"key": "val_loss", "label": "Validation loss", "color": "kept", "axis": "right"}], "bar", note="This is where precision and recall stop being the whole story: a filter can be right about every document it removed and still leave too little text to train on.")
R.table("results", "Results", [{"key": "corpus", "label": "Corpus"}, {"key": "docs", "label": "Documents", "fmt": "int"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "val_loss", "label": "Validation loss", "fmt": "num"}], results)
if best["corpus"] != mine["corpus"]:
    R.note(f"Your filtered corpus was not the best of the {len(results)} — {best['corpus']} was. Compare the token counts before changing a threshold: the usual cause is a filter that removed too much rather than the wrong things.")
R.output("val_loss", mine["val_loss"]).output("clean", I["clean"]).output("f1", I.get("f1")).output("data_source", I.get("data_source", "generated")).output("tokenizer", str(tok_path)).output("tokenizer_label", tk["label"])
R.save()
