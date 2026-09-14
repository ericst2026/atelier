"""Grader: does the filter agree with a held-out judgement, and does it train a better model?"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import hf
from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import HFTokenizer
from atelier_mini.train import pretrain

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

    # held-out documents the guided steps never load: the tail of each source
    raw = hf.dataset_split("curation", "c4-raw", "train", limit=150000)[-25000:]
    good = hf.dataset_split("curation", "fineweb-edu", "train", limit=150000)[-8000:]
    progress(10, f"{len(raw):,} raw and {len(good):,} reference documents held out")
    kept_raw = [d for d in raw if mod.keep(d)]
    kept_good = sum(1 for d in good if mod.keep(d))
    recall = kept_good / max(len(good), 1)
    keep_rate = len(kept_raw) / max(len(raw), 1)
    test("keeps good text", recall >= 0.8, f"kept {recall:.1%} of the already-filtered documents", 25, 25 * min(1.0, recall / 0.9))
    test("removes something", 0.05 <= keep_rate <= 0.9, f"kept {keep_rate:.1%} of raw web text", 15, 15 if 0.05 <= keep_rate <= 0.9 else 0)
    R.metric("recall", "Good text kept", recall, "pct", "kept").metric("keep_rate", "Raw text kept", keep_rate, "pct", "raw")
    if len(kept_raw) < 200:
        raise ValueError("the filter kept too little to train on")

    progress(35, "training a fixed model on what you kept")
    tok = HFTokenizer(hf.model_path("gpt2") / "tokenizer.json")
    val_rows = [d["text"] for d in hf.dataset_split("pretrain", "wikitext-103", "validation", limit=3000) if d.get("text")]
    vs = pack(val_rows, tok, run_dir / "val.bin", max_tokens=400_000)
    ts = pack([d["text"] for d in kept_raw if d.get("text")], tok, run_dir / "train.bin", max_tokens=5_000_000)
    cfg = MiniConfig.preset("tiny", tok.vocab_size)
    cfg.block_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    model = MiniLM(cfg).to(device)
    res = pretrain(model, TokenStream(run_dir / "train.bin", ts["dtype"]), TokenStream(run_dir / "val.bin", vs["dtype"]), run_dir / "m",
                   max_iters=600, batch_size=32, block_size=cfg.block_size, lr=8e-4, warmup=50, eval_every=100, device=device,
                   on_log=lambda r: progress(40 + 55 * r["step"] / 600, f"step {r['step']}/600" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else "")))
    vl = res["best_val_loss"]
    # 50 points: nothing at a loss of 6.0, full marks at 4.2 or below on this tiny budget
    test("trains a better model", vl < 5.5, f"validation loss {vl:.3f} after 600 steps on 5M tokens", 50, 50 * min(1.0, max(0.0, (6.0 - vl) / 1.8)))
    R.metric("val_loss", "Validation loss", vl, "num", "hold").metric("tokens", "Tokens kept", ts["tokens"], "int", "sky")
except Exception as exc:
    test("filter", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
