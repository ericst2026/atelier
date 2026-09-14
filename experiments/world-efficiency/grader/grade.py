"""Grader: accuracy against speed, with the memory it took."""
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

N = 150
TOKENS = 128
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
    spec = importlib.util.spec_from_file_location("student_optimise", project / "project" / "optimise.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "optimise", None)), "optimise() found", 10, 10)

    ckpt = project / "outputs" / "model.pt"
    model, ck = MiniLM.load(ckpt, device)
    tok = MiniTokenizer.load(ck.get("tokenizer") or (project / "outputs" / "tokenizer.json"))
    world = World(lang=ck.get("lang", "en"), seed=88)
    system = ck.get("system", world.system_prompt)
    tasks = world.eval_set(N, seed=161_803_398)
    prompts = [t["prompt"] for t in tasks]

    progress(10, "baseline")
    base_model, _ = MiniLM.load(ckpt, device)
    base_bytes = sum(p.numel() * p.element_size() for p in base_model.parameters())
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    base_out = [g[0] for g in generate(base_model, tok, prompts, TOKENS, 0.0, batch_size=16, system=system)]
    if device == "cuda":
        torch.cuda.synchronize()
    base_secs = time.time() - t0
    base_acc = sum(world.grade(g, t["answer"]) for g, t in zip(base_out, tasks)) / N
    del base_model
    torch.cuda.empty_cache()

    progress(45, "your version")
    try:
        engine = mod.optimise(model, tok, system)
    except TypeError:
        engine = mod.optimise(model, tok)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = engine.generate(prompts, TOKENS)
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    acc = sum(world.grade(g, t["answer"]) for g, t in zip(out, tasks)) / N
    tps = N * TOKENS / max(secs, 1e-9)
    base_tps = N * TOKENS / max(base_secs, 1e-9)
    speedup = tps / base_tps
    mem = sum(b.numel() * b.element_size() for b in engine.model.buffers()) + sum(p.numel() * p.element_size() for p in engine.model.parameters()) if hasattr(engine, "model") else base_bytes

    test("accuracy kept", acc >= base_acc - 0.05, f"{acc:.1%} against the baseline {base_acc:.1%}", 45, 45 * min(1.0, max(0.0, 1 - max(0.0, base_acc - acc) / 0.15)))
    test("faster", speedup > 1.1, f"{speedup:.2f}× ({tps:,.0f} against {base_tps:,.0f} tokens/s)", 35, 35 * min(1.0, max(0.0, (speedup - 1.0) / 2.0)))
    test("smaller", mem <= base_bytes, f"{mem / 1e6:.0f} MB against {base_bytes / 1e6:.0f} MB", 10, 10 * min(1.0, max(0.0, (base_bytes - mem) / (base_bytes * 0.5))))
    R.metric("accuracy", "Accuracy", acc, "pct", "kept").metric("baseline_accuracy", "Baseline", base_acc, "pct", "raw")
    R.metric("tokens_per_sec", "Tokens per second", tps, "num", "sky").metric("speedup", "Speedup", speedup, "num", "hold").metric("memory_mb", "Weights in memory", mem / 1e6, "num", "raw")
    R.chart("compare", "Against the baseline", [{"which": "baseline", "tokens_per_sec": base_tps, "accuracy": base_acc}, {"which": "yours", "tokens_per_sec": tps, "accuracy": acc}], "which", [{"key": "tokens_per_sec", "label": "Tokens/s", "color": "kept"}, {"key": "accuracy", "label": "Accuracy", "color": "sky", "axis": "right"}], "bar")
except Exception as exc:
    test("optimise", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
