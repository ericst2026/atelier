"""Grader: does the student's harness agree with the reference, item by item?"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_score import option_logprobs, pick  # noqa: E402
from lib_tasks import load  # noqa: E402

MODEL = "SmolLM2-135M"
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
    spec = importlib.util.spec_from_file_location("student_harness", project / "project" / "harness.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "evaluate", None)), "evaluate() found", 10, 10)

    mp = hf.model_path(MODEL)
    tok = hf.load_tokenizer(mp, padding_side="right")
    model = hf.load_model(mp)
    # held-out items, a different seed from anything the guided steps used
    items = load("arc-easy", 120, seed=90_210) + load("hellaswag", 80, seed=90_210)
    progress(20, f"{len(items)} held-out multiple-choice items")
    theirs = mod.evaluate(model, tok, items, "choice")
    if len(theirs) != len(items):
        raise ValueError(f"evaluate() returned {len(theirs)} results for {len(items)} items")
    reference = []
    for i, x in enumerate(items):
        scores = option_logprobs(model, tok, x["question"], x["choices"], 8)
        reference.append(pick(scores, "mean") == x["answer"])
        if i % 25 == 0:
            progress(30 + 45 * i / len(items), f"reference {i}/{len(items)}")
    agree = sum(1 for t, r in zip(theirs, reference) if bool(t.get("correct")) == r) / len(items)
    test("agrees with the reference", agree >= 0.9, f"{agree:.1%} of items scored the same way", 45, 45 * min(1.0, max(0.0, (agree - 0.5) / 0.45)))
    R.metric("agreement", "Agreement with the reference", agree, "pct", "kept")
    their_acc = sum(1 for t in theirs if t.get("correct")) / len(items)
    ref_acc = sum(reference) / len(items)
    test("plausible accuracy", abs(their_acc - ref_acc) < 0.1, f"your harness reports {their_acc:.1%}, the reference {ref_acc:.1%}", 20, 20 * min(1.0, max(0.0, 1 - abs(their_acc - ref_acc) / 0.2)))
    R.metric("accuracy", "Accuracy your harness reports", their_acc, "pct", "sky").metric("reference_accuracy", "Reference", ref_acc, "pct", "hold")

    progress(80, "generation task")
    gen_items = load("gsm8k", 40, seed=90_210)
    gen_res = mod.evaluate(model, tok, gen_items, "generation")
    ok = isinstance(gen_res, list) and len(gen_res) == len(gen_items) and all("correct" in g for g in gen_res)
    test("handles generation too", ok, f"returned {len(gen_res) if isinstance(gen_res, list) else 0} results, {sum(1 for g in gen_res if g.get('correct')) if ok else 0} scored correct", 25, 25 if ok else 0)
except Exception as exc:
    test("harness", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
