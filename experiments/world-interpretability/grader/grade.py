"""Grader: does the probe generalise, and are the induction heads the right ones?"""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, parse_args, progress
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import fit_probe, hidden_states, induction_scores, probe_accuracy  # noqa: E402

N = 600
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
    spec = importlib.util.spec_from_file_location("student_interp", project / "project" / "interp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    ok = all(callable(getattr(mod, f, None)) for f in ("train_probe", "predict_probe", "induction_heads"))
    test("interface", ok, "train_probe(), predict_probe() and induction_heads() found", 10, 10 if ok else 0)

    from atelier_mini.model import MiniLM
    from atelier_mini.tok import MiniTokenizer

    model, ck = MiniLM.load(project / "outputs" / "model.pt", "cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tok = MiniTokenizer.load(ck.get("tokenizer") or (project / "outputs" / "tokenizer.json"))
    # a seed the guided steps never use
    world = World(lang=ck.get("lang", "en"), seed=1)
    tasks = world.eval_set(N, seed=553_311_007)
    fams = sorted({t["family"] for t in tasks})
    y = np.array([fams.index(t["family"]) for t in tasks])
    progress(10, f"reading hidden states for {N} prompts")
    H = hidden_states(model, tok, [t["prompt"] for t in tasks], "last", batch_size=16, progress=lambda d, t: progress(10 + 30 * d / t, f"{d}/{t}"))
    layer = H.shape[1] // 2
    cut = int(N * 0.7)
    Xtr, Xte, ytr, yte = H[:cut, layer, :], H[cut:, layer, :], y[:cut], y[cut:]
    majority = float(np.bincount(yte).max() / len(yte))

    probe = mod.train_probe(Xtr, ytr)
    pred = np.asarray(mod.predict_probe(probe, Xte)).reshape(-1)
    acc = float((pred == yte).mean())
    reference = probe_accuracy(fit_probe(Xtr, ytr, len(fams)), Xte, yte)
    test("probe beats the baseline", acc > majority + 0.1, f"{acc:.1%} against {majority:.1%} for guessing the commonest label", 30, 30 * min(1.0, max(0.0, (acc - majority) / max(reference - majority, 1e-6))))
    test("probe matches the reference", acc >= reference - 0.08, f"reference logistic regression gets {reference:.1%}", 20, 20 * min(1.0, max(0.0, 1 - max(0.0, reference - acc) / 0.2)))
    R.metric("probe_accuracy", "Probe accuracy", acc, "pct", "kept").metric("reference", "Reference probe", reference, "pct", "hold").metric("majority", "Commonest label", majority, "pct", "raw")

    progress(55, "finding induction heads")
    theirs = [tuple(int(x) for x in pair) for pair in mod.induction_heads(model, tok)][:5]
    ref_scores = induction_scores(model, tok, 48, 16, seed=11, progress=lambda d, t: progress(60 + 30 * d / t, f"reference {d}/{t}"))
    ranked = sorted(((l, h, float(ref_scores[l, h])) for l in range(ref_scores.shape[0]) for h in range(ref_scores.shape[1])), key=lambda r: -r[2])
    ref_top5 = {(l, h) for l, h, _ in ranked[:5]}
    ref_top10 = {(l, h) for l, h, _ in ranked[:10]}
    overlap5 = len(set(theirs) & ref_top5) / 5
    overlap10 = len(set(theirs) & ref_top10) / 5
    test("finds the induction heads", overlap10 >= 0.6, f"{len(set(theirs) & ref_top5)}/5 in the reference top five, {len(set(theirs) & ref_top10)}/5 in the top ten", 40, 40 * (0.6 * overlap5 + 0.4 * overlap10))
    R.metric("head_overlap", "Agreement on the top five", overlap5, "pct", "sky")
    R.table("heads", "Your heads against the reference", [{"key": "rank", "label": "Rank"}, {"key": "yours", "label": "You found"}, {"key": "reference", "label": "Reference"}, {"key": "score", "label": "Reference score", "fmt": "num"}], [{"rank": i + 1, "yours": f"{theirs[i][0]}.{theirs[i][1]}" if i < len(theirs) else "—", "reference": f"{ranked[i][0]}.{ranked[i][1]}", "score": ranked[i][2]} for i in range(5)])
except Exception as exc:
    test("interpretability", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
