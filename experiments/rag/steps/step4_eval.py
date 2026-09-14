"""Step 4 — wrong passages, and knowing when to refuse."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf
from atelier_mini.embed import Embedder, Index

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_qa import SYSTEM, build_prompt, exact_match, f1, is_refusal  # noqa: E402

parse_args()
P = params({"n_unanswerable": 100, "max_new_tokens": 64})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
meta = json.loads(Path(I["index_meta"]).read_text())
questions = hf.read_jsonl(I["questions"], limit=300)
mp = hf.model_path(I["reader"])
tok = hf.load_tokenizer(mp)
model = hf.load_model(mp)
k = int(I.get("k", 3))


def ask(prompts):
    return [g[0].strip().split("\n")[0] for g in hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), 0.0, batch_size=16)]


def gold_present(q, passages) -> bool:
    ctx = " ".join((q.get("context") or "").lower().split())
    return any(" ".join(p["text"].lower().split())[:120] in ctx for p in passages)


# 1. how accuracy depends on whether the right passage was retrieved
progress(10, "answering with the retrieved passages")
prompts = [hf.chat_prompt(tok, build_prompt(q["question"], q["retrieved"][:k]), SYSTEM) for q in questions]
preds = ask(prompts)
golds = [(q.get("answers") or {}).get("text") or [] for q in questions]
found = [gold_present(q, q["retrieved"][:k]) for q in questions]
hit_em = [exact_match(p, g) for p, g in zip(preds, golds)]
when_found = [e for e, f in zip(hit_em, found) if f]
when_missed = [e for e, f in zip(hit_em, found) if not f]
refused_when_missed = [is_refusal(p) for p, f in zip(preds, found) if not f]

# 2. questions whose answer is not in the corpus at all
progress(45, "asking questions with no answer in the corpus")
unans = hf.read_jsonl(I["unanswerable"], limit=int(P["n_unanswerable"])) if I.get("unanswerable") else []
unans_refusal = None
if unans:
    emb = Embedder(hf.model_path(meta["embedder"]))
    chunks = hf.read_jsonl(I["chunks"])
    index = Index(np.load(I["vectors"]), chunks)
    qv = emb.encode([q["question"] for q in unans], batch_size=128, prefix=meta.get("query_prefix", ""))
    res = index.search(qv, k)
    u_prompts = [hf.chat_prompt(tok, build_prompt(q["question"], r), SYSTEM) for q, r in zip(unans, res)]
    u_preds = ask(u_prompts)
    unans_refusal = sum(is_refusal(p) for p in u_preds) / len(u_preds)
    u_rows = [{"question": q["question"][:160], "pred": p[:180], "ok": "✓" if is_refusal(p) else "✗"} for q, p in list(zip(unans, u_preds))[:15]]
progress(85, "scoring")

R = Result()
R.metric("exact_match", "Exact match", sum(hit_em) / len(questions), "pct", "kept")
R.metric("when_found", "When the right passage was retrieved", sum(when_found) / max(len(when_found), 1), "pct", "sky", help=f"{len(when_found)} questions")
R.metric("when_missed", "When it was not", sum(when_missed) / max(len(when_missed), 1), "pct", "dup", help=f"{len(when_missed)} questions — anything correct here the model already knew")
if unans_refusal is not None:
    R.metric("refusal", "Refused when there is no answer", unans_refusal, "pct", "kept" if unans_refusal > 0.5 else "dup", help=f"{len(unans)} questions with no answer in the corpus")
R.chart("split", "Accuracy split by whether retrieval worked", [{"case": "right passage retrieved", "accuracy": sum(when_found) / max(len(when_found), 1)}, {"case": "wrong passages", "accuracy": sum(when_missed) / max(len(when_missed), 1)}], "case", [{"key": "accuracy", "label": "Exact match", "color": "kept"}], "bar", y_domain=[0, 1], note="The right bar is the model answering from memory or from luck. A system that scores well there and badly on refusal is guessing confidently.")
R.chart("behaviour", "What it does when the passage is wrong", [{"behaviour": "answers correctly anyway", "share": sum(when_missed) / max(len(when_missed), 1)}, {"behaviour": "says there is no answer", "share": sum(refused_when_missed) / max(len(refused_when_missed), 1)}, {"behaviour": "answers wrongly", "share": 1 - (sum(when_missed) + sum(refused_when_missed)) / max(len(when_missed), 1)}], "behaviour", [{"key": "share", "label": "Share", "color": "hold"}], "bar", y_domain=[0, 1])
R.table("failures", "Wrong answers with the right passage present", [{"key": "question", "label": "Question"}, {"key": "gold", "label": "Reference"}, {"key": "pred", "label": "The model said"}], [{"question": q["question"][:150], "gold": ", ".join(g[:2])[:80], "pred": p[:180]} for q, p, g, e, f in zip(questions, preds, golds, hit_em, found) if f and not e][:15], note="These are the reader's mistakes, not the retriever's — the passage was there.")
if unans:
    R.table("unanswerable", "Questions with no answer in the corpus", [{"key": "ok", "label": ""}, {"key": "question", "label": "Question"}, {"key": "pred", "label": "The model said"}], u_rows)
R.output("exact_match", sum(hit_em) / len(questions)).output("refusal", unans_refusal)
R.save()
