"""Grader: load outputs/ckpt.pt through project/model.py, check the parameter budget, measure val loss."""
import importlib.util
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import hf

BUDGET = 30_000_000
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.0f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


try:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = importlib.util.spec_from_file_location("student_model", project / "project" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "build_model", None)), "build_model found", 10, 10)
    ck = torch.load(project / "outputs" / "ckpt.pt", map_location=device)
    model = mod.build_model(ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    ok = n <= BUDGET
    test("budget", ok, f"{n:,} parameters (budget {BUDGET:,})", 20, 20 if ok else 0)
    progress(30, "checkpoint loaded")
    # fixed validation slice: the grader's own GPT-2 tokenized TinyStories val set
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(hf.model_path("gpt2") / "tokenizer.json"))
    rows = hf.dataset_split("pretrain", "tinystories", "validation", limit=2000) if (hf.MATERIALS / "datasets/pretrain/tinystories/validation.jsonl").exists() else hf.dataset_split("pretrain", "tinystories", "train", limit=2000)
    ids = []
    for r in rows:
        ids += tk.encode(r["text"]).ids + [tk.token_to_id("<|endoftext|>")]
    data = np.array(ids, dtype=np.int64)
    T = int(ck["config"].get("block_size", 256))
    torch.manual_seed(0)
    losses = []
    for b in range(40):
        ix = torch.randint(len(data) - T - 1, (16,))
        x = torch.stack([torch.from_numpy(data[i : i + T]) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T]) for i in ix]).to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    vl = float(np.mean(losses))
    # 70 points: 0 at loss 4.0 (unigram-ish), 70 at loss ≤ 1.6
    pts = 70 * min(1.0, max(0.0, (4.0 - vl) / 2.4))
    test("val_loss", vl < 3.0, f"validation loss {vl:.3f} (ppl {math.exp(vl):.1f})", 70, pts)
    R.metric("val_loss", "Validation loss", vl, "num", "hold").metric("perplexity", "Perplexity", math.exp(vl), "num", "hold").metric("params", "Parameters", n, "int", "kept")
except Exception as exc:
    test("load", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
