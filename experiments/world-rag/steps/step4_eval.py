"""Step 4 — recall, hybrid search, and answering from what was found."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.embed import BM25, Index, MiniEmbedder, recall_at_k
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"k_values": ["1", "3", "5", "10", "20"], "hybrid_weight": 0.7, "n_answer": 200, "passages": 2})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
docs = read_jsonl(I["documents"])
queries = read_jsonl(I["queries"])
texts = [d["text"] for d in docs]
golds = [q["gold_id"] for q in queries]
ks = sorted(int(k) for k in P["k_values"])
world = World(lang=I.get("lang", "en"), seed=1)

ck = torch.load(I["embedder"], map_location=device, weights_only=False)
from atelier_mini.model import MiniConfig  # noqa: E402

model = MiniLM(MiniConfig(**ck["config"])).to(device)
model.load_state_dict(ck["model_state"])
tok = MiniTokenizer.load(ck.get("tokenizer") or I["tokenizer"])
emb = MiniEmbedder(model, tok, layer=ck["layer"], dim=ck["dim"] if ck["dim"] != model.config.n_embd else None, max_length=ck["max_length"]).to(device)
if ck.get("proj") and isinstance(emb.proj, torch.nn.Linear):
    emb.proj.load_state_dict(ck["proj"])

progress(8, "encoding the library")
D = emb.encode(texts, batch_size=256, progress=lambda d, t: progress(8 + 22 * d / t, f"{d}/{t}"))
Q = emb.encode([q["query"] for q in queries], batch_size=256)
index = Index(D, [{"doc_id": i} for i in range(len(texts))])
dense = index.search(Q, max(ks))
progress(40, "keyword search")
bm = BM25(texts)
lexical = [bm.search(q["query"], max(ks)) for q in queries]

# hybrid: reciprocal rank fusion, which needs no score calibration between the two
w = float(P["hybrid_weight"])
hybrid = []
for dres, lres in zip(dense, lexical):
    scores = {}
    for rank, r in enumerate(dres):
        scores[r["doc_id"]] = scores.get(r["doc_id"], 0.0) + w / (60 + rank)
    for rank, (idx, _) in enumerate(lres):
        scores[idx] = scores.get(idx, 0.0) + (1 - w) / (60 + rank)
    hybrid.append([{"doc_id": i} for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])[: max(ks)]])

methods = {
    "BM25": [[{"doc_id": i} for i, _ in r] for r in lexical],
    "trained embedder": dense,
    "hybrid": hybrid,
}
curve = {}
for name, res in methods.items():
    for k in ks:
        curve.setdefault(k, {"k": k})[name] = recall_at_k(res, "doc_id", golds, k)
rows = [curve[k] for k in ks]
best_name = max(methods, key=lambda n: curve[5][n] if 5 in curve else curve[ks[-1]][n])

answers = None
if int(P["n_answer"]) > 0 and I.get("reader_run_run"):
    ro = I["reader_run_run"]["outputs"]
    reader, rck = MiniLM.load(ro["sft_model"], device)
    rtok = MiniTokenizer.load(ro["tokenizer"])
    system = ro.get("system") or rck.get("system") or world.system_prompt
    n = min(int(P["n_answer"]), len(queries))
    npass = int(P["passages"])
    conditions = {}
    for ci, (label, getter) in enumerate((
        ("no context", lambda i: []),
        (f"retrieved ({best_name})", lambda i: [texts[r["doc_id"]] for r in methods[best_name][i][:npass]]),
        ("the correct document", lambda i: [texts[golds[i]]]),
    )):
        prompts = []
        for i in range(n):
            ctx = getter(i)
            body = ("\n".join(ctx) + "\n\n") if ctx else ""
            prompts.append(f"{body}{queries[i]['query']}")
        gens = generate(reader, rtok, prompts, 48, 0.0, batch_size=32, system=system, progress=lambda d, t: progress(55 + 40 * (ci + d / t) / 3, f"{label}: {d}/{t}"))
        correct = [world.grade(g[0], queries[i]["answer"]) for i, g in enumerate(gens)]
        conditions[label] = {"accuracy": sum(correct) / n, "gens": [g[0] for g in gens], "correct": correct}
    answers = conditions
    del reader
    torch.cuda.empty_cache()

R = Result()
R.metric("recall5", f"Recall@5 · {best_name}", curve[5][best_name] if 5 in curve else curve[ks[-1]][best_name], "pct", "kept")
R.metric("bm25", "Recall@5 · BM25", curve[5]["BM25"] if 5 in curve else curve[ks[-1]]["BM25"], "pct", "hold")
R.metric("dense", "Recall@5 · trained embedder", curve[5]["trained embedder"] if 5 in curve else curve[ks[-1]]["trained embedder"], "pct", "sky")
if answers:
    key = f"retrieved ({best_name})"
    R.metric("answer_accuracy", "Answers correct with retrieval", answers[key]["accuracy"], "pct", "kept", help=f"no context: {answers['no context']['accuracy']:.1%}; with the correct document: {answers['the correct document']['accuracy']:.1%}")
R.chart("recall", "Recall against k", rows, "k", [{"key": n, "label": n} for n in methods], "line", y_domain=[0, 1], note="The hybrid usually wins because the two methods fail on different queries: keyword search on paraphrase, the embedder on rare exact strings like a number.")
if answers:
    R.chart("answers", "Answering, by what the reader was given", [{"condition": k, "accuracy": v["accuracy"]} for k, v in answers.items()], "condition", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1], note="The right-hand bar is what the reader could do with perfect retrieval. The distance to the middle bar belongs to the retriever; the distance from the right bar to 100% belongs to the reader.")
R.table("recall", "Recall", [{"key": "k", "label": "k"}] + [{"key": n, "label": n, "fmt": "pct"} for n in methods], rows)
if answers:
    key = f"retrieved ({best_name})"
    R.table("examples", "End to end", [{"key": "ok", "label": ""}, {"key": "query", "label": "Query"}, {"key": "gold", "label": "Answer"}, {"key": "retrieved", "label": "Top document retrieved"}, {"key": "said", "label": "The reader said"}], [{"ok": "✓" if answers[key]["correct"][i] else "✗", "query": queries[i]["query"][:150], "gold": queries[i]["answer"], "retrieved": texts[methods[best_name][i][0]["doc_id"]][:140], "said": answers[key]["gens"][i][:140]} for i in range(min(15, len(queries)))])
R.output("recall5", curve[5][best_name] if 5 in curve else curve[ks[-1]][best_name]).output("embedder", I["embedder"]).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en"))
if answers:
    R.output("answer_accuracy", answers[f"retrieved ({best_name})"]["accuracy"])
R.save()
