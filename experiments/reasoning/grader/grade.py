"""Grader: accuracy + token efficiency of project/solve.py on hidden GSM8K problems."""
import importlib.util
import os
import random
import sys
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import hf

MODEL = "Qwen2.5-0.5B-Instruct"
N = 150
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
    spec = importlib.util.spec_from_file_location("student_solve", project / "project" / "solve.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    mp = hf.model_path(MODEL)
    tok, model = hf.load_tokenizer(mp), hf.load_model(mp)
    meter = {"tokens": 0}

    def generate(prompt, n=1, temperature=0.0, max_new_tokens=384):
        outs = hf.generate_batch(model, tok, [hf.chat_prompt(tok, prompt)], min(int(max_new_tokens), 1024), float(temperature), num_return_sequences=max(1, min(int(n), 32)))[0]
        meter["tokens"] += sum(len(tok(o).input_ids) for o in outs)
        return outs

    rows = hf.dataset_split("rl", "gsm8k", "test")
    random.Random(2024).shuffle(rows)
    hidden = rows[:N]
    correct, errors = 0, 0
    for i, r in enumerate(hidden):
        try:
            pred = mod.solve(r["question"], generate)
            correct += hf.answers_equal(str(pred), hf.gsm8k_gold(r["answer"]))
        except Exception:
            errors += 1
            if errors > 10:
                raise
        if i % 10 == 0:
            progress(5 + 90 * i / N, f"{i}/{N} · {correct} correct")
    acc = correct / N
    tpp = meter["tokens"] / N
    eff = 30 * min(1.0, max(0.0, (3000 - tpp) / 2600))
    test("accuracy", acc >= 0.3, f"{correct}/{N} hidden problems ({100 * acc:.1f}%), {errors} solver errors", 70, 70 * acc)
    test("efficiency", tpp <= 1200, f"{tpp:.0f} generated tokens per problem", 30, eff)
    R.metric("accuracy", "Hidden accuracy", acc, "pct", "kept").metric("tokens_per_problem", "Tokens per problem", tpp, "num", "sky")
except Exception as exc:
    test("solve", False, f"{type(exc).__name__}: {exc}", 100, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
