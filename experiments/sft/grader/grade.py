"""Grader: hidden held-out loss + judge preference vs the base model."""
import os
import re
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import hf

BASE = "Qwen2.5-0.5B"
JUDGE = "Qwen2.5-1.5B-Instruct"
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

    adapter = project / "outputs" / "adapter"
    full = project / "outputs" / "model"
    if (adapter / "adapter_config.json").exists():
        tuned = hf.load_model(hf.model_path(BASE), adapter=adapter)
        test("artifact", True, "outputs/adapter found", 10, 10)
    elif (full / "config.json").exists():
        tuned = hf.load_model(full)
        test("artifact", True, "outputs/model found", 10, 10)
    else:
        raise FileNotFoundError("outputs/adapter or outputs/model missing — run project/train.py first")
    tok = hf.load_tokenizer(hf.model_path(BASE))
    # hidden validation: last 300 rows of alpaca-cleaned (not the rows students see first)
    rows = hf.dataset_split("sft", "alpaca-cleaned", "train")
    hidden = [{"messages": [{"role": "user", "content": r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")}, {"role": "assistant", "content": r["output"]}]} for r in rows[-300:]]
    tuned_loss = hf.eval_loss(tuned, tok, hidden[:150])
    progress(35, f"tuned loss {tuned_loss:.3f}")
    base = hf.load_model(hf.model_path(BASE))
    base_loss = hf.eval_loss(base, tok, hidden[:150])
    progress(50, f"base loss {base_loss:.3f}")
    gain = base_loss - tuned_loss
    pts = 40 * min(1.0, max(0.0, gain / 0.6))
    test("held_out_loss", gain > 0, f"loss {tuned_loss:.3f} vs base {base_loss:.3f}", 40, pts)
    R.metric("tuned_loss", "Held-out loss", tuned_loss, "num", "kept").metric("base_loss", "Base loss", base_loss, "num", "raw")
    prompts_msgs = [r["messages"][:-1] for r in hidden[150:180]]
    prompts = [hf.chat_prompt(tok, m[-1]["content"]) for m in prompts_msgs]
    a = [g[0] for g in hf.generate_batch(base, tok, prompts, 200, 0.0)]
    b = [g[0] for g in hf.generate_batch(tuned, tok, prompts, 200, 0.0)]
    del base, tuned
    torch.cuda.empty_cache()
    jt = hf.load_tokenizer(hf.model_path(JUDGE))
    judge = hf.load_model(hf.model_path(JUDGE))
    jp = [hf.chat_prompt(jt, f"Reply with exactly one word: A, B, or TIE.\n\nRequest:\n{m[-1]['content'][:1200]}\n\nAnswer A:\n{a[i][:1200]}\n\nAnswer B:\n{b[i][:1200]}\n\nWhich answer is more helpful and correct?") for i, m in enumerate(prompts_msgs)]
    v = [re.findall(r"\b(A|B|TIE)\b", g[0].upper()) for g in hf.generate_batch(judge, jt, jp, 6, 0.0)]
    wins = sum(1 for x in v if x and x[0] == "B")
    win_rate = wins / len(v)
    test("judge", win_rate >= 0.5, f"judge prefers your model on {wins}/{len(v)} prompts", 50, 50 * min(1.0, win_rate / 0.8))
    R.metric("win_rate", "Judge win rate", win_rate, "pct", "kept")
except Exception as exc:
    test("load", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
