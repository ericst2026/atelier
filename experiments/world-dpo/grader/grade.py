"""Grader: accuracy on unseen questions, plus a check that the model can still write.

DPO's characteristic failure is a policy that ranks well and produces badly, so an
empty or truncated answer costs more here than it does after fine-tuning."""
import os
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


try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = project / "outputs" / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError("outputs/model.pt missing — run project/train.py first")
    model, ck = MiniLM.load(ckpt, device)
    test("artifact", True, f"outputs/model.pt loaded, {model.num_params():,} parameters", 10, 10)
    tok_path = ck.get("tokenizer") or (project / "outputs" / "tokenizer.json")
    if not Path(tok_path).exists():
        raise FileNotFoundError("the checkpoint does not record a tokenizer path, and outputs/tokenizer.json is missing")
    tok = MiniTokenizer.load(tok_path)
    world = World(lang=ck.get("lang", "en"), seed=1)
    system = ck.get("system", world.system_prompt)
    # a seed the guided steps never use
    tasks = world.eval_set(N, seed=13_579_246)
    progress(10, f"asking {N} unseen questions")
    gens = generate(model, tok, [t["prompt"] for t in tasks], 192, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(10 + 55 * d / t, f"{d}/{t}"))
    correct = [world.grade(g[0], t["answer"]) for g, t in zip(gens, tasks)]
    acc = sum(correct) / N
    by_family = {}
    for c, t in zip(correct, tasks):
        f = by_family.setdefault(t["family"], [0, 0])
        f[0] += c
        f[1] += 1
    test("accuracy", acc >= 0.3, f"{sum(correct)}/{N} correct ({acc:.1%})", 50, 50 * min(1.0, acc / 0.8))
    R.metric("accuracy", "Accuracy", acc, "pct", "kept")
    empty = sum(1 for g in gens if len(g[0].strip()) < 3) / N
    healthy = sum(1 for g in gens if world.answer_prefix in g[0]) / N
    test("still writes", empty < 0.05 and healthy > 0.8, f"{empty:.1%} empty, {healthy:.0%} produced an answer line", 10, 10 * max(0.0, min(1.0, healthy - empty)))
    R.metric("empty", "Empty answers", empty, "pct", "dup")
    R.chart("families", "Accuracy by family", [{"family": f, "accuracy": v[0] / v[1]} for f, v in sorted(by_family.items())], "family", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
    R.table("answers", "A sample of what it said", [{"key": "ok", "label": ""}, {"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "model", "label": "The model said"}], [{"ok": "✓" if correct[i] else "✗", "family": tasks[i]["family"], "question": tasks[i]["prompt"][:200], "gold": tasks[i]["answer"], "model": gens[i][0][:220]} for i in range(20)])

    base_path = ck.get("base_model")
    if base_path and Path(base_path).exists():
        base, _ = MiniLM.load(base_path, device)
        bg = generate(base, tok, [t["prompt"] for t in tasks[:150]], 192, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(70 + 25 * d / t, f"base {d}/{t}"))
        base_acc = sum(world.grade(g[0], t["answer"]) for g, t in zip(bg, tasks[:150])) / 150
        gain = acc - base_acc
        test("improvement", gain > 0, f"{gain:+.1%} over the base model ({base_acc:.1%})", 30, 30 * min(1.0, max(0.0, gain / 0.4)))
        R.metric("base_accuracy", "Base model", base_acc, "pct", "raw").metric("gain", "Gained", gain, "pct", "sky")
    else:
        # no base recorded: award on absolute accuracy alone rather than punishing
        test("improvement", True, "no base model recorded in the checkpoint; scored on accuracy alone", 30, 30 * min(1.0, acc / 0.8))
except Exception as exc:
    test("load", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
