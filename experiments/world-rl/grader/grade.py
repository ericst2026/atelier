"""Grader: is the reward sane, and did the policy actually get better?"""
import importlib.util
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

N = 300
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


world = World(lang="en", seed=5)
probe = world.eval_set(60, seed=404_101)
try:
    spec = importlib.util.spec_from_file_location("student_reward", project / "project" / "reward.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    prefers, resists = 0, 0
    for t in probe:
        good = "\n".join(t["steps"]) + f"\n{world.answer_prefix} {t['answer']}"
        wrong = "\n".join(t["steps"]) + f"\n{world.answer_prefix} {t['answer']}0"
        empty = ""
        padded = (f"{world.answer_prefix} {t['answer']} " + "the " * 400)
        rg = float(mod.reward(t, good))
        if rg > float(mod.reward(t, wrong)):
            prefers += 1
        if rg >= float(mod.reward(t, empty)) and rg >= float(mod.reward(t, padded)):
            resists += 1
    test("prefers correct", prefers >= 57, f"a correct answer scored higher in {prefers}/60 cases", 15, 15 * prefers / 60)
    test("not trivially gamed", resists >= 57, f"an empty or padded completion did not outscore a correct one in {resists}/60 cases", 15, 15 * resists / 60)
except Exception as exc:
    test("reward", False, f"{type(exc).__name__}: {exc}", 30, 0)
    traceback.print_exc()
progress(20, "reward checked")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = project / "outputs" / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError("outputs/model.pt missing — run project/train.py first")
    model, ck = MiniLM.load(ckpt, device)
    tok_path = ck.get("tokenizer") or (project / "outputs" / "tokenizer.json")
    tok = MiniTokenizer.load(tok_path)
    w = World(lang=ck.get("lang", "en"), seed=5)
    system = ck.get("system", w.system_prompt)
    tasks = w.eval_set(N, seed=61_803_398)
    gens = generate(model, tok, [t["prompt"] for t in tasks], 192, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(25 + 45 * d / t, f"{d}/{t}"))
    correct = [w.grade(g[0], t["answer"]) for g, t in zip(gens, tasks)]
    acc = sum(correct) / N
    test("accuracy", acc >= 0.3, f"{sum(correct)}/{N} correct ({acc:.1%})", 45, 45 * min(1.0, acc / 0.8))
    R.metric("accuracy", "Accuracy", acc, "pct", "kept")
    marked_wrong = sum(1 for g, c in zip(gens, correct) if not c and w.answer_prefix in g[0]) / N
    R.metric("wrong_marked", "Confidently wrong", marked_wrong, "pct", "dup")

    base_path = ck.get("base_model")
    if base_path and Path(base_path).exists():
        base, _ = MiniLM.load(base_path, device)
        bg = generate(base, tok, [t["prompt"] for t in tasks[:150]], 192, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(75 + 20 * d / t, f"base {d}/{t}"))
        base_acc = sum(w.grade(g[0], t["answer"]) for g, t in zip(bg, tasks[:150])) / 150
        gain = acc - base_acc
        test("improvement", gain > 0, f"{gain:+.1%} over the policy you started from ({base_acc:.1%})", 25, 25 * min(1.0, max(0.0, gain / 0.15)))
        R.metric("start_accuracy", "Starting policy", base_acc, "pct", "raw").metric("gain", "Gained", gain, "pct", "sky")
    else:
        test("improvement", True, "no starting policy recorded; scored on accuracy alone", 25, 25 * min(1.0, acc / 0.8))
except Exception as exc:
    test("policy", False, f"{type(exc).__name__}: {exc}", 70, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
