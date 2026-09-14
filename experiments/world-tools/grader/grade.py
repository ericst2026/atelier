"""Grader: accuracy through a tool loop, plus whether the calls are well formed and used."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_tools import RESULT_CLOSE, RESULT_OPEN, find_call, run_call  # noqa: E402

N = 250
MAX_TURNS = 2
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
    spec = importlib.util.spec_from_file_location("student_tools", project / "project" / "tools.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    tools = dict(getattr(mod, "TOOLS", {}))
    test("interface", bool(tools), f"{len(tools)} tool(s): {', '.join(sorted(tools))}", 10, 10 if tools else 0)

    model, ck = MiniLM.load(project / "outputs" / "model.pt", device)
    tok = MiniTokenizer.load(ck.get("tokenizer") or (project / "outputs" / "tokenizer.json"))
    world = World(lang=ck.get("lang", "en"), seed=33)
    system = ck.get("system", world.system_prompt)
    tasks = world.eval_set(N, seed=112_358_132)

    transcripts = [""] * N
    called = valid = 0
    active = list(range(N))
    for turn in range(MAX_TURNS + 1):
        if not active:
            break
        prompts = [tasks[i]["prompt"] + ("\n" + transcripts[i] if transcripts[i] else "") for i in active]
        gens = generate(model, tok, prompts, 128, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(15 + 70 * (turn + d / t) / (MAX_TURNS + 1), f"turn {turn}: {d}/{t}"))
        still = []
        for idx, g in zip(active, gens):
            transcripts[idx] += g[0]
            call = find_call(g[0], tools)
            if call and turn < MAX_TURNS:
                called += 1
                result, ok = run_call(tools, *call)
                valid += ok
                transcripts[idx] += RESULT_OPEN + result + RESULT_CLOSE + "\n"
                still.append(idx)
        active = still

    correct = sum(world.grade(t, task["answer"]) for t, task in zip(transcripts, tasks))
    acc = correct / N
    valid_rate = valid / max(called, 1)
    with_result = [t for t in transcripts if RESULT_OPEN in t]
    used = sum(1 for t in with_result if (lambda r, tail: r and r in tail)(t.split(RESULT_OPEN)[1].split(RESULT_CLOSE)[0], t.split(RESULT_CLOSE, 1)[1] if RESULT_CLOSE in t else ""))
    used_rate = used / max(len(with_result), 1)

    test("accuracy", acc >= 0.3, f"{correct}/{N} correct ({acc:.1%})", 55, 55 * min(1.0, acc / 0.8))
    test("well-formed calls", valid_rate >= 0.9, f"{valid}/{called} calls parsed ({valid_rate:.1%})", 20, 20 * valid_rate if called else 0)
    test("uses the result", used_rate >= 0.7, f"the returned value appears in the answer {used_rate:.1%} of the time", 15, 15 * used_rate if with_result else 0)
    R.metric("accuracy", "Accuracy", acc, "pct", "kept").metric("valid_calls", "Calls that parsed", valid_rate, "pct", "sky").metric("used", "Results used", used_rate, "pct", "hold").metric("call_rate", "Calls per question", called / N, "num", "raw")
    R.table("transcripts", "A sample", [{"key": "question", "label": "Question"}, {"key": "transcript", "label": "What happened"}], [{"question": tasks[i]["prompt"][:140], "transcript": transcripts[i][:300]} for i in range(15)])
except Exception as exc:
    test("tools", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
