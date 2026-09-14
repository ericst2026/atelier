"""Grader: accuracy divided by the number of weights it took to get it."""
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

N = 250
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = importlib.util.spec_from_file_location("student_adapt", project / "project" / "adapt.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "adapt", None)), "adapt() found", 10, 10)

    model, ck = MiniLM.load(project / "outputs" / "model.pt", device)
    tok = MiniTokenizer.load(ck.get("tokenizer") or (project / "outputs" / "tokenizer.json"))
    world = World(lang=ck.get("lang", "en"), seed=15)
    system = ck.get("system", world.system_prompt)
    trainable = int(ck.get("trainable") or json.loads((project / "outputs" / "census.json").read_text())["trainable"])
    total = sum(p.numel() for p in model.parameters())
    test("census", 0 < trainable <= total, f"{trainable:,} trainable of {total:,} ({trainable / total:.2%})", 10, 10 if 0 < trainable <= total else 0)

    tasks = world.eval_set(N, seed=141_421_356)
    progress(15, f"{N} unseen questions")
    gens = generate(model, tok, [t["prompt"] for t in tasks], 192, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(15 + 75 * d / t, f"{d}/{t}"))
    correct = sum(world.grade(g[0], t["answer"]) for g, t in zip(gens, tasks))
    acc = correct / N
    per_million = acc / max(trainable / 1e6, 1e-9)
    test("accuracy", acc >= 0.3, f"{correct}/{N} correct ({acc:.1%})", 50, 50 * min(1.0, acc / 0.8))
    # full marks for reaching a useful accuracy on under a million trained weights
    test("efficiency", per_million >= 0.1, f"{per_million:.3f} accuracy per million trained weights", 30, 30 * min(1.0, per_million / 0.5))
    R.metric("accuracy", "Accuracy", acc, "pct", "kept")
    R.metric("trainable", "Trainable parameters", trainable, "int", "sky")
    R.metric("accuracy_per_million", "Accuracy per million trained weights", per_million, "num", "hold")
except Exception as exc:
    test("adapt", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
