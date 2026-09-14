"""Step 3 — answer with no context, with retrieved context, and with the correct passage."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_qa import SYSTEM, build_prompt, exact_match, f1, is_refusal  # noqa: E402

parse_args()
P = params({"reader": "Qwen2.5-0.5B-Instruct", "n_questions": 250, "k": 3, "max_new_tokens": 64, "conditions": ["none", "retrieved", "gold"]})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
questions = hf.read_jsonl(I["questions"], limit=int(P["n_questions"]))
mp = hf.model_path(P["reader"])
tok = hf.load_tokenizer(mp)
model = hf.load_model(mp)
k = int(P["k"])

conditions = [c for c in P["conditions"] if c in ("none", "retrieved", "gold")] or ["retrieved"]
results, rows = [], []
for ci, cond in enumerate(conditions):
    prompts = []
    for q in questions:
        if cond == "none":
            passages = []
        elif cond == "gold":
            passages = [{"text": (q.get("context") or "")[:1500]}]
        else:
            passages = q["retrieved"][:k]
        prompts.append(hf.chat_prompt(tok, build_prompt(q["question"], passages), SYSTEM))
    gens = hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), 0.0, batch_size=16, progress=lambda d, t: progress(5 + 90 * (ci + d / t) / len(conditions), f"{cond}: {d}/{t}"))
    preds = [g[0].strip().split("\n")[0] for g in gens]
    golds = [(q.get("answers") or {}).get("text") or [] for q in questions]
    em = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(questions)
    f1s = sum(f1(p, g) for p, g in zip(preds, golds)) / len(questions)
    refusals = sum(is_refusal(p) for p in preds) / len(questions)
    results.append({"condition": cond, "exact_match": em, "f1": f1s, "refusal": refusals, "mean_chars": sum(len(p) for p in preds) / len(questions)})
    if cond == conditions[-1]:
        rows = [{"question": q["question"][:150], "gold": ", ".join(g[:2])[:80], "pred": p[:150], "ok": "✓" if exact_match(p, g) else "✗"} for q, p, g in list(zip(questions, preds, golds))[:20]]
    hf.write_jsonl(run_dir / f"answers_{cond}.jsonl", [{"question": q["question"], "gold": g, "pred": p, "correct": exact_match(p, g)} for q, p, g in zip(questions, preds, golds)])

by = {r["condition"]: r for r in results}
R = Result()
if "retrieved" in by:
    R.metric("with_context", "Exact match with retrieval", by["retrieved"]["exact_match"], "pct", "kept")
if "none" in by:
    R.metric("without_context", "Exact match with no context", by["none"]["exact_match"], "pct", "raw", help="what the model knows without being told")
if "gold" in by:
    R.metric("ceiling", "Exact match with the correct passage", by["gold"]["exact_match"], "pct", "hold", help="the reader's ceiling; the gap to your retrieval score belongs to the retriever")
R.metric("f1", "Token overlap (F1)", by.get("retrieved", results[-1])["f1"], "num", "sky", help="partial credit for nearly-right answers")
R.chart("conditions", "Accuracy by condition", [{"condition": r["condition"], "exact_match": r["exact_match"], "f1": r["f1"]} for r in results], "condition", [{"key": "exact_match", "label": "Exact match", "color": "kept"}, {"key": "f1", "label": "F1", "color": "sky"}], "bar", y_domain=[0, 1], note="No context shows what the model memorised. The correct passage shows what it can read. Retrieval sits between them, and where it sits tells you which half to work on.")
R.chart("refusal", "How often it declined to answer", [{"condition": r["condition"], "refusal": r["refusal"]} for r in results], "condition", [{"key": "refusal", "label": "Refused", "color": "dup"}], "bar", y_domain=[0, 1])
R.table("answers", "A sample", [{"key": "ok", "label": ""}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Reference"}, {"key": "pred", "label": "The model said"}], rows)
R.output("reader", P["reader"]).output("with_context", by.get("retrieved", results[-1])["exact_match"]).output("k", k)
for key in ("questions", "unanswerable", "vectors", "chunks", "index_meta", "embedder", "recall_5"):
    if key in I:
        R.output(key, I[key])
R.save()
