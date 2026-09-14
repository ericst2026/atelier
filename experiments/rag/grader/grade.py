"""Grader: recall, exact match, and whether it refuses when it should."""
import importlib.util
import os
import sys
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_qa import exact_match, f1, is_refusal  # noqa: E402

N = 150
ARTICLES = 8000
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


def load(name):
    spec = importlib.util.spec_from_file_location(f"student_{name}", project / "project" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    sys.path.insert(0, str(project / "project"))
    retriever = load("retriever")
    reader = load("answer")
    test("interface", all(callable(getattr(m, f, None)) for m, f in ((retriever, "build"), (retriever, "search"), (reader, "answer"))), "build(), search() and answer() found", 10, 10)

    corpus = hf.dataset_split("rag", "wikipedia-simple", "train", limit=ARTICLES)
    progress(10, f"building the index over {len(corpus):,} articles")
    index = retriever.build(corpus)

    squad = hf.dataset_split("rag", "squad-v2", "validation", limit=8000)
    answerable = [q for q in squad if (q.get("answers") or {}).get("text")][-N:]
    unanswerable = [q for q in squad if not (q.get("answers") or {}).get("text")][-60:]
    mp = hf.model_path("Qwen2.5-0.5B-Instruct")
    tok, model = hf.load_tokenizer(mp), hf.load_model(mp)

    def generate(prompt, system=None, max_new_tokens=64):
        return hf.generate_batch(model, tok, [hf.chat_prompt(tok, prompt, system)], min(int(max_new_tokens), 128), 0.0)[0]

    def gold_hit(passages, context):
        ctx = " ".join((context or "").lower().split())
        return any(" ".join((p.get("text") or "").lower().split())[:120] in ctx for p in passages)

    recall_hits, em, f1s, sizes = 0, 0, 0.0, []
    for i, q in enumerate(answerable):
        passages = retriever.search(index, q["question"], 5)
        sizes.append(len(passages))
        if gold_hit(passages, q.get("context")):
            recall_hits += 1
        pred = reader.answer(q["question"], passages[:3], generate)
        em += exact_match(pred, q["answers"]["text"])
        f1s += f1(pred, q["answers"]["text"])
        if i % 10 == 0:
            progress(30 + 45 * i / len(answerable), f"{i}/{len(answerable)} · {em} correct")
    recall = recall_hits / len(answerable)
    em_rate = em / len(answerable)
    mean_passages = sum(sizes) / max(len(sizes), 1)
    test("returns a shortlist", mean_passages <= 10, f"{mean_passages:.1f} passages returned per question", 10, 10 if mean_passages <= 10 else 0)
    test("recall@5", recall >= 0.4, f"the answering passage was in the top 5 for {recall:.1%} of questions", 35, 35 * min(1.0, recall / 0.8))
    test("exact match", em_rate >= 0.2, f"{em}/{len(answerable)} exact ({em_rate:.1%}), F1 {f1s / len(answerable):.2f}", 30, 30 * min(1.0, em_rate / 0.55))
    R.metric("recall_5", "Recall@5", recall, "pct", "sky").metric("exact_match", "Exact match", em_rate, "pct", "kept").metric("f1", "F1", f1s / len(answerable), "num", "hold")

    progress(80, "questions with no answer in the corpus")
    refused = 0
    for q in unanswerable:
        passages = retriever.search(index, q["question"], 5)
        refused += is_refusal(reader.answer(q["question"], passages[:3], generate))
    refusal = refused / len(unanswerable)
    test("refuses when it should", refusal >= 0.4, f"said there was no answer for {refusal:.1%} of unanswerable questions", 15, 15 * min(1.0, refusal / 0.7))
    R.metric("refusal", "Refused when unanswerable", refusal, "pct", "raw")
except Exception as exc:
    test("run", False, f"{type(exc).__name__}: {exc}", 0, 0)
    traceback.print_exc()

score = round(min(100, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.output("score", score)
R.save()
