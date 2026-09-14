"""Grader: labels, loss, and whether enough text survived to train on."""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import pretrain
from atelier_nlp import bpe
from atelier_world import World, make_corpus, score_filter

N_DOCS = 20000
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


try:
    spec = importlib.util.spec_from_file_location("student_filters", project / "project" / "filters.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "keep", None)), "keep() found", 10, 10)

    # a seed the guided steps never use
    world = World(lang="en", seed=17)
    docs = make_corpus(world, N_DOCS, seed=602_214_076)
    progress(10, f"{len(docs):,} documents, labels withheld")
    kept_flags = [bool(mod.keep({"text": d["text"]})) for d in docs]
    s = score_filter(docs, kept_flags)
    test("labels", s["f1"] >= 0.6, f"precision {s['precision']:.2f}, recall {s['recall']:.2f}, F1 {s['f1']:.2f}", 40, 40 * min(1.0, max(0.0, (s["f1"] - 0.3) / 0.6)))
    R.metric("f1", "F1", s["f1"], "num", "kept").metric("precision", "Precision", s["precision"], "pct", "sky").metric("recall", "Recall", s["recall"], "pct", "hold")
    R.chart("by_kind", "Kept, by kind", [{"kind": k, "kept": v, "target": 1.0 if k == "clean" else 0.0} for k, v in s["by_kind"].items()], "kind", [{"key": "kept", "label": "Kept", "color": "raw"}, {"key": "target", "label": "Should keep", "color": "kept"}], "bar", y_domain=[0, 1])

    kept_texts = [d["text"] for d, k in zip(docs, kept_flags) if k]
    test("enough to train on", len(kept_texts) >= 500, f"{len(kept_texts):,} documents survived", 10, 10 if len(kept_texts) >= 500 else 0)
    if len(kept_texts) < 200:
        raise ValueError("too little text left to train anything")

    progress(35, "training a fixed model on what you kept")
    tokenizer = bpe.train(bpe.count_words(d["text"] for d in docs), "byte", 2048, 2)
    tok_path = run_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    tok = MiniTokenizer.load(tok_path)
    val_texts = [d["text"] for d in world.documents(2000, seed=90_001)]
    vs = pack(val_texts, tok, run_dir / "val.bin", max_tokens=800_000)
    ts = pack(kept_texts, tok, run_dir / "train.bin", max_tokens=8_000_000)
    cfg = MiniConfig.preset("tiny", tok.vocab_size)
    cfg.block_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    model = MiniLM(cfg).to(device)
    res = pretrain(model, TokenStream(run_dir / "train.bin", ts["dtype"]), TokenStream(run_dir / "val.bin", vs["dtype"]), run_dir / "m",
                   max_iters=500, batch_size=32, block_size=cfg.block_size, lr=8e-4, warmup=50, eval_every=100, device=device,
                   on_log=lambda r: progress(40 + 55 * r["step"] / 500, f"step {r['step']}/500" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else "")))
    vl = res["best_val_loss"]
    test("trains a better model", vl < 4.5, f"validation loss {vl:.3f} on {ts['tokens']:,} tokens", 40, 40 * min(1.0, max(0.0, (5.2 - vl) / 1.6)))
    R.metric("val_loss", "Validation loss", vl, "num", "hold").metric("tokens", "Tokens kept", ts["tokens"], "int", "raw")
except Exception as exc:
    test("filter", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
