"""Grader: accuracy and token efficiency of project/solve.py on unseen questions."""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_mini.gen import generate as sample
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

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
    ckpt = project / "outputs" / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError("outputs/model.pt missing — save the model your solver should use")
    model, ck = MiniLM.load(ckpt, device)
    tok = MiniTokenizer.load(ck.get("tokenizer") or (project / "outputs" / "tokenizer.json"))
    world = World(lang=ck.get("lang", "en"), seed=3)
    system = ck.get("system", world.system_prompt)
    spec = importlib.util.spec_from_file_location("student_solve", project / "project" / "solve.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "solve", None)), "solve() found", 10, 10)

    meter = {"tokens": 0}

    def generate(prompt, n=1, temperature=0.0, max_new_tokens=192):
        outs = sample(model, tok, [str(prompt)], min(int(max_new_tokens), 512), float(temperature), num_samples=max(1, min(int(n), 16)), batch_size=16, system=system)[0]
        meter["tokens"] += sum(len(tok.encode(o)) for o in outs)
        return outs

    tasks = world.eval_set(N, seed=271_828_182)
    correct, errors = 0, 0
    for i, t in enumerate(tasks):
        try:
            pred = mod.solve(dict(t, lang=ck.get("lang", "en")), generate)
            correct += world.grade(f"{world.answer_prefix} {pred}", t["answer"])
        except Exception:
            errors += 1
            if errors > 10:
                raise
        if i % 10 == 0:
            progress(10 + 85 * i / N, f"{i}/{N} · {correct} correct")
    acc = correct / N
    tpp = meter["tokens"] / N
    test("accuracy", acc >= 0.3, f"{correct}/{N} correct ({acc:.1%}), {errors} solver errors", 70, 70 * acc)
    test("efficiency", tpp <= 1200, f"{tpp:.0f} generated tokens per question", 30, 30 * min(1.0, max(0.0, (3000 - tpp) / 2600)))
    R.metric("accuracy", "Accuracy", acc, "pct", "kept").metric("tokens_per_question", "Tokens per question", tpp, "num", "sky")
except Exception as exc:
    test("solve", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
