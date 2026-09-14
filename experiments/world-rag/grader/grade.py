"""Grader: recall on a library the student has never seen, and the time it takes."""
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from atelier_sdk import Result, parse_args, progress
from atelier_mini.embed import BM25
from atelier_world import Library, World

N_DOCS = 6000
N_QUERIES = 400
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
    sys.path.insert(0, str(project / "project"))
    spec = importlib.util.spec_from_file_location("student_retriever", project / "project" / "retriever.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok = callable(getattr(mod, "build", None)) and callable(getattr(mod, "search", None))
    test("interface", ok, "build() and search() found", 10, 10 if ok else 0)

    # if the student trained an embedder, hand it to them the way their own script does
    ckpt = project / "outputs" / "embedder.pt"
    if ckpt.exists() and hasattr(mod, "load_embedder"):
        try:
            mod._state["embedder"] = mod.load_embedder(str(ckpt), "cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:
            print(f"[grader] could not load outputs/embedder.pt: {exc}")

    # a library from a seed the guided steps never use
    world = World(lang="en", seed=4242)
    lib = Library(world, n_docs=N_DOCS, seed=31_337, cluster_size=4)
    texts = [d["text"] for d in lib.documents]
    queries = lib.queries(N_QUERIES, seed=808_080)
    golds = [q["gold_id"] for q in queries]
    progress(15, f"{len(texts):,} documents, {len(queries)} queries")

    t0 = time.time()
    index = mod.build(texts)
    build_secs = time.time() - t0
    progress(45, f"index built in {build_secs:.1f}s")
    t0 = time.time()
    results = mod.search(index, [q["query"] for q in queries], 5)
    search_ms = (time.time() - t0) / len(queries) * 1000
    progress(75, f"{search_ms:.1f} ms per query")

    if not isinstance(results, list) or len(results) != len(queries):
        raise ValueError(f"search() returned {type(results).__name__} of length {len(results) if isinstance(results, list) else '?'} for {len(queries)} queries")
    sizes = [len(r) for r in results]
    test("returns a shortlist", max(sizes) <= 10, f"{sum(sizes) / len(sizes):.1f} documents returned per query", 10, 10 if max(sizes) <= 10 else 0)
    r5 = sum(1 for r, g in zip(results, golds) if g in list(r)[:5]) / len(queries)
    r1 = sum(1 for r, g in zip(results, golds) if list(r)[:1] == [g]) / len(queries)

    bm = BM25(texts)
    baseline = sum(1 for q in queries if q["gold_id"] in [i for i, _ in bm.search(q["query"], 5)]) / len(queries)
    test("beats keyword search", r5 > baseline + 0.05, f"recall@5 {r5:.1%} against BM25's {baseline:.1%}", 45, 45 * min(1.0, max(0.0, (r5 - baseline) / (0.95 - baseline))))
    test("ranks it first", r1 >= 0.3, f"recall@1 {r1:.1%}", 25, 25 * min(1.0, r1 / 0.75))
    test("fast enough", search_ms < 50, f"{search_ms:.1f} ms per query, index built in {build_secs:.0f}s", 10, 10 * min(1.0, max(0.0, (100 - search_ms) / 95)))
    R.metric("recall5", "Recall@5", r5, "pct", "kept").metric("recall1", "Recall@1", r1, "pct", "sky")
    R.metric("bm25", "BM25 recall@5", baseline, "pct", "hold").metric("search_ms", "Milliseconds per query", search_ms, "num", "raw")
    R.chart("compare", "Against keyword search", [{"method": "BM25", "recall": baseline}, {"method": "yours", "recall": r5}], "method", [{"key": "recall", "label": "Recall@5", "color": "kept"}], "bar", y_domain=[0, 1])
    R.table("misses", "Queries you missed", [{"key": "query", "label": "Query"}, {"key": "gold", "label": "The right document"}, {"key": "got", "label": "Your first result"}], [{"query": queries[i]["query"][:160], "gold": texts[golds[i]][:140], "got": texts[list(results[i])[0]][:140] if len(results[i]) else "—"} for i in range(len(queries)) if golds[i] not in list(results[i])[:5]][:15])
except Exception as exc:
    test("retriever", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
