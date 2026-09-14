"""Grader: reward sanity + hidden GSM8K accuracy of outputs/policy."""
import importlib.util
import os
import random
import re
import sys
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import hf

args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.0f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


gsm = hf.dataset_split("rl", "gsm8k", "test")
rng = random.Random(99)
rng.shuffle(gsm)
hidden = gsm[-250:]
try:
    spec = importlib.util.spec_from_file_location("student_reward", project / "project" / "reward.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    ok = 0
    for r in hidden[:60]:
        gold = hf.gsm8k_gold(r["answer"])
        good = r["answer"]
        wrong = re.sub(r"####.*", f"#### {float(gold) + 7 if gold.replace('.', '', 1).lstrip('-').isdigit() else '0'}", r["answer"])
        if mod.reward(r["question"], good, gold) > mod.reward(r["question"], wrong, gold):
            ok += 1
    rate = ok / 60
    test("reward_sanity", rate >= 0.95, f"reward prefers the reference solution in {ok}/60 cases", 20, 20 * rate)
except Exception as exc:
    test("reward_sanity", False, f"{type(exc).__name__}: {exc}", 20, 0)
    traceback.print_exc()
progress(20, "reward checked")
try:
    policy = project / "outputs" / "policy"
    if not (policy / "config.json").exists():
        raise FileNotFoundError("outputs/policy missing — run project/train.py first")
    tok = hf.load_tokenizer(policy)
    model = hf.load_model(policy)
    SYSTEM = "Solve the problem step by step, then give the final answer on its own line as '#### <number>'."
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True) for r in hidden[60:]]
    gens = hf.generate_batch(model, tok, prompts, 384, 0.0, progress=lambda d, t: progress(25 + 70 * d / t, f"{d}/{t}"))
    correct = sum(1 for g, r in zip(gens, hidden[60:]) if hf.answers_equal(hf.extract_answer(g[0]), hf.gsm8k_gold(r["answer"])))
    acc = correct / len(prompts)
    pts = 80 * min(1.0, max(0.0, (acc - 0.15) / 0.45))
    test("accuracy", acc >= 0.3, f"{correct}/{len(prompts)} hidden problems correct ({100 * acc:.1f}%)", 80, pts)
    R.metric("accuracy", "Hidden accuracy", acc, "pct", "kept")
    flagged = sum(1 for g in gens if hf.extract_answer(g[0]) is None or len(g[0]) > 2500)
    R.metric("flagged", "Flagged completions", flagged, "int", "dup", help="no answer or very long")
except Exception as exc:
    test("accuracy", False, f"{type(exc).__name__}: {exc}", 80, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
