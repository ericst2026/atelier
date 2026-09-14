"""Grader: parameter budget and validation loss on a freshly generated stream."""
import importlib.util
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from atelier_sdk import Result, parse_args, progress
from atelier_mini.data import pack
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

BUDGET = 30_000_000
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


try:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = importlib.util.spec_from_file_location("student_model", project / "project" / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "build_model", None)), "build_model found", 10, 10)

    ck = torch.load(project / "outputs" / "model.pt", map_location=device, weights_only=False)
    model = mod.build_model(ck["config"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    ok = n <= BUDGET
    test("budget", ok, f"{n:,} parameters against a budget of {BUDGET:,}", 20, 20 if ok else 0)
    progress(25, "checkpoint loaded")

    # The grader packs its own stream: same world, a seed nobody has trained on,
    # and the tokenizer the student's checkpoint says it was trained with.
    tok_path = ck.get("tokenizer") or (project / "outputs" / "tokenizer.json")
    if not Path(tok_path).exists():
        raise FileNotFoundError("save the tokenizer path in the checkpoint, or copy tokenizer.json into outputs/")
    tok = MiniTokenizer.load(tok_path)
    if tok.vocab_size != ck["config"]["vocab_size"]:
        raise ValueError(f"the checkpoint expects a vocabulary of {ck['config']['vocab_size']}, the tokenizer gives {tok.vocab_size}")
    world = World(lang=ck.get("lang", "en"), seed=808_017)
    run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
    stats = pack((d["text"] for d in world.documents(20000)), tok, run_dir / "grader_val.bin", max_tokens=2_000_000)
    data = np.fromfile(run_dir / "grader_val.bin", dtype=np.uint16 if stats["dtype"] == "uint16" else np.uint32)
    progress(55, f"packed {data.size:,} held-out tokens")

    T = int(ck["config"]["block_size"])
    torch.manual_seed(0)
    losses = []
    for b in range(60):
        ix = torch.randint(len(data) - T - 1, (16,))
        x = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64)) for i in ix]).to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(float(loss))
        if b % 15 == 0:
            progress(55 + 40 * b / 60, f"batch {b}/60")
    vl = float(np.mean(losses))
    # 70 points: nothing at a loss of 4.0, full marks at 1.2 or below
    pts = 70 * min(1.0, max(0.0, (4.0 - vl) / 2.8))
    test("val_loss", vl < 3.0, f"validation loss {vl:.3f} (perplexity {math.exp(min(vl, 20)):.1f})", 70, pts)
    R.metric("val_loss", "Validation loss", vl, "num", "hold").metric("perplexity", "Perplexity", math.exp(min(vl, 20)), "num", "hold").metric("params", "Parameters", n, "int", "kept")
except Exception as exc:
    test("load", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
