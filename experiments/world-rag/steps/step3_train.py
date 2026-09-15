"""Step 3 — contrastive training."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.embed import Index, recall_at_k, train_contrastive

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_embed import make_embedder, n_blocks, save_embedder  # noqa: E402

parse_args()
P = params({"layer": -1, "dim": 0, "epochs": 3.0, "batch_size": 64, "lr": 1e-4, "temperature": 0.05, "hard_negatives": True, "freeze_body": False})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
docs = read_jsonl(I["documents"])
queries = read_jsonl(I["queries"])
pairs = read_jsonl(I["pairs"])
texts = [d["text"] for d in docs]
fmt = I.get("model_format", "atelier")
layer = int(P["layer"])
if layer < 0:
    layer = int(I.get("best_layer") or -1)
emb = make_embedder(fmt, I["base_model"], I.get("tokenizer"), device, layer=layer, dim=int(P["dim"]) or None)
if fmt == "hf" and not 0 <= emb.layer < n_blocks(emb):
    emb.layer = -1
split = max(64, len(pairs) // 20)
val, train = pairs[:split], pairs[split:]
steps_total = max(1, int(len(train) * float(P["epochs"]) / int(P["batch_size"])))
progress(3, f"{len(train):,} pairs · batch {P['batch_size']} · {steps_total} steps")


def recall5(e) -> float:
    D = e.encode(texts, batch_size=256)
    Q = e.encode([q["query"] for q in queries], batch_size=256)
    res = Index(D, [{"doc_id": i} for i in range(len(texts))]).search(Q, 5)
    return recall_at_k(res, "doc_id", [q["gold_id"] for q in queries], 5)


before = recall5(emb)
progress(10, f"before training: recall@5 {before:.1%}")
res = train_contrastive(emb, train, run_dir, val, epochs=float(P["epochs"]), batch_size=int(P["batch_size"]), lr=float(P["lr"]), temperature=float(P["temperature"]), hard_negatives=bool(P["hard_negatives"]), freeze_body=bool(P["freeze_body"]), device=device,
                        on_log=lambda r: progress(10 + 80 * r["step"] / steps_total, f"step {r['step']}/{steps_total}" + (f" · in-batch {r['in_batch_accuracy']:.0%}" if "in_batch_accuracy" in r else f" · loss {r.get('loss', 0):.3f}"), step=r["step"], **{k: v for k, v in r.items() if k in ("loss", "in_batch_accuracy")}))
after = recall5(emb)
# a HuggingFace base is stored as its fine-tuned weights plus where the base (config, tokenizer) lives
save_embedder(emb, run_dir / "embedder.pt", {"tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "base_model": I["base_model"]})

hist = res["history"]
evals = [h for h in hist if "in_batch_accuracy" in h]
bm25 = float(I.get("bm25_recall5") or 0)
R = Result()
R.metric("recall5", "Recall@5 after training", after, "pct", "kept", help=f"before: {before:.1%}; keyword search: {bm25:.1%}")
if fmt == "hf":
    R.note(f"The encoder started from {I.get('model_label', 'a HuggingFace model')}; its weights were trained contrastively here the same way as an Atelier model's.")
R.metric("in_batch_accuracy", "In-batch accuracy", evals[-1]["in_batch_accuracy"] if evals else None, "pct", "sky", help=f"against {int(P['batch_size']) - 1} negatives drawn from the same batch")
R.metric("gain", "Gained over keyword search", after - bm25, "pct", "kept" if after > bm25 else "dup")
R.metric("dim", "Embedding dimension", emb.dim, "int", "hold", help=f"index size for {len(texts):,} documents: {len(texts) * emb.dim * 4 / 1e6:.1f} MB")
R.chart("progress", "In-batch accuracy during training", [{"step": h["step"], "accuracy": h["in_batch_accuracy"]} for h in evals], "step", [{"key": "accuracy", "label": "Ranks its own document first", "color": "kept"}], "line", y_domain=[0, 1], note="This saturates well before recall does: picking the right document out of 64 is much easier than out of the whole library.")
R.chart("loss", "Loss", [{"step": h["step"], "loss": h["loss"]} for h in hist if "loss" in h], "step", [{"key": "loss", "label": "InfoNCE", "color": "hold"}], "line")
R.chart("recall", "Where the retriever stands", [{"method": "BM25", "recall": bm25}, {"method": "zero-shot", "recall": float(I.get("zero_shot_recall5") or before)}, {"method": "trained", "recall": after}], "method", [{"key": "recall", "label": "Recall@5", "color": "kept"}], "bar", y_domain=[0, 1])
R.artifact(run_dir / "embedder.pt", "embedder.pt")
R.output("embedder", str(run_dir / "embedder.pt")).output("recall5", after).output("layer", emb.layer).output("dim", emb.dim)
for k in ("documents", "queries", "pairs", "library", "base_model", "tokenizer", "lang", "bm25_recall5", "zero_shot_recall5", "model_format", "model_label", "data_source", "data_label"):
    if k in I:
        R.output(k, I[k])
R.save()
