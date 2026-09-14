"""Grader: needle at 2048 across ten depths, and no damage to short inputs."""
import importlib.util
import os
import random
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
from lib_needle import make_haystack, needle_prompt, plant  # noqa: E402

LENGTH = 2048
DEPTHS = [i / 9 for i in range(10)]
PER_CELL = 6
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
    spec = importlib.util.spec_from_file_location("student_extend", project / "project" / "extend.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    test("interface", callable(getattr(mod, "extend", None)), "extend() found", 10, 10)

    model, ck = MiniLM.load(project / "outputs" / "model.pt", device)
    tok = MiniTokenizer.load(ck.get("tokenizer") or (project / "outputs" / "tokenizer.json"))
    world = World(lang=ck.get("lang", "en"), seed=77)
    if model.config.block_size < LENGTH:
        mod.extend(model, LENGTH)
    rng = random.Random(2718)

    grid, hits = [], 0
    for di, d in enumerate(DEPTHS):
        needles = [plant(world, tok, LENGTH - 60, d, rng) for _ in range(PER_CELL)]
        gens = generate(model, tok, [needle_prompt(n["document"], n["question"], world.answer_prefix) for n in needles], 24, 0.0, batch_size=4)
        found = sum(1 for g, n in zip(gens, needles) if n["answer"] in g[0])
        hits += found
        grid.append({"depth": f"{int(d * 100)}%", "found": found / PER_CELL})
        progress(10 + 60 * (di + 1) / len(DEPTHS), f"depth {d:.0%}: {found}/{PER_CELL}")
    acc = hits / (len(DEPTHS) * PER_CELL)
    test("needle", acc >= 0.3, f"found the planted fact in {hits}/{len(DEPTHS) * PER_CELL} documents ({acc:.1%})", 60, 60 * min(1.0, acc / 0.8))
    R.metric("needle_accuracy", "Needle found", acc, "pct", "kept")
    R.chart("depths", "Found by depth", grid, "depth", [{"key": "found", "label": "Found", "color": "kept"}], "line", y_domain=[0, 1])

    progress(75, "checking it is still good at short inputs")
    short_losses = []
    for _ in range(30):
        text, _ = make_haystack(world, tok, 256, rng)
        ids = torch.tensor([tok.encode(text, bos=True)[:257]], device=device)
        if ids.shape[1] > 16:
            with torch.no_grad():
                _, loss = model(ids[:, :-1], ids[:, 1:])
            short_losses.append(float(loss))
    short = sum(short_losses) / max(len(short_losses), 1)
    base_path = ck.get("base_model")
    if base_path and Path(base_path).exists():
        base, _ = MiniLM.load(base_path, device)
        base_losses = []
        rng2 = random.Random(2718)
        for _ in range(30):
            text, _ = make_haystack(world, tok, 256, rng2)
            ids = torch.tensor([tok.encode(text, bos=True)[:257]], device=device)
            if ids.shape[1] > 16:
                with torch.no_grad():
                    _, loss = base(ids[:, :-1], ids[:, 1:])
                base_losses.append(float(loss))
        base_short = sum(base_losses) / max(len(base_losses), 1)
        regress = (short - base_short) / max(base_short, 1e-9)
        test("no damage when short", regress < 0.1, f"short-input loss {short:.3f} against {base_short:.3f} before ({regress:+.1%})", 30, 30 * min(1.0, max(0.0, (0.2 - regress) / 0.2)))
        R.metric("short_loss", "Short-input loss", short, "num", "hold").metric("regression", "Change when short", regress, "pct", "dup" if regress > 0.05 else "kept")
    else:
        test("no damage when short", True, f"no base model recorded; short-input loss {short:.3f}", 30, 20)
        R.metric("short_loss", "Short-input loss", short, "num", "hold")
except Exception as exc:
    test("extend", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
