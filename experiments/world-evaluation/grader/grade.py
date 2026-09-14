"""Grader: does the harness agree with a reference, item by item?"""
import importlib.util
import os
import random
import sys
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World, multiple_choice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_harness import option_logprobs, pick  # noqa: E402

N = 200
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
    spec = importlib.util.spec_from_file_location("student_harness", project / "project" / "harness.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "evaluate", None)), "evaluate() found", 10, 10)

    model, ck = MiniLM.load(project / "outputs" / "model.pt", device)
    model.eval()
    tok = MiniTokenizer.load(ck.get("tokenizer") or (project / "outputs" / "tokenizer.json"))
    world = World(lang=ck.get("lang", "en"), seed=1)
    rng = random.Random(31_415)
    items = [multiple_choice(world, it, rng, 4) for it in world.eval_set(N, seed=27_182_818)]
    progress(15, f"{len(items)} held-out items")

    theirs = mod.evaluate(model, tok, items, "choice")
    if not isinstance(theirs, list) or len(theirs) != len(items):
        raise ValueError(f"evaluate() returned {len(theirs) if isinstance(theirs, list) else type(theirs).__name__} for {len(items)} items")
    reference = []
    for i, x in enumerate(items):
        scores = option_logprobs(model, tok, x["question"], x["options"], 16)
        reference.append(pick(scores, "mean") == x["answer"])
        if i % 25 == 0:
            progress(35 + 40 * i / len(items), f"reference {i}/{len(items)}")
    agree = sum(1 for t, r in zip(theirs, reference) if bool(t.get("correct")) == r) / len(items)
    test("agrees with the reference", agree >= 0.9, f"{agree:.1%} of items scored the same way", 45, 45 * min(1.0, max(0.0, (agree - 0.5) / 0.45)))
    R.metric("agreement", "Agreement", agree, "pct", "kept")

    their_acc = sum(1 for t in theirs if t.get("correct")) / len(items)
    ref_acc = sum(reference) / len(items)
    test("plausible accuracy", abs(their_acc - ref_acc) < 0.1, f"your harness reports {their_acc:.1%}, the reference {ref_acc:.1%}", 20, 20 * min(1.0, max(0.0, 1 - abs(their_acc - ref_acc) / 0.2)))
    R.metric("accuracy", "Your harness reports", their_acc, "pct", "sky").metric("reference_accuracy", "Reference", ref_acc, "pct", "hold")

    progress(80, "open questions")
    open_items = [{"question": it["prompt"], "answer": it["answer"], "family": it["family"]} for it in world.eval_set(40, seed=27_182_818)]
    gen_res = mod.evaluate(model, tok, open_items, "generation")
    ok = isinstance(gen_res, list) and len(gen_res) == len(open_items) and all("correct" in g for g in gen_res)
    test("handles generation too", ok, f"returned {len(gen_res) if isinstance(gen_res, list) else 0} results", 25, 25 if ok else 0)
except Exception as exc:
    test("harness", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
