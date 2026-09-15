"""Step 2 — the language model as an embedder, before any training."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.embed import Index, recall_at_k

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_embed import make_embedder, n_blocks as count_blocks, width  # noqa: E402

parse_args()
P = params({"layers": ["-1"], "sweep_all_layers": True, "max_length": 128, "k_values": ["1", "5", "20"]})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
docs = read_jsonl(I["documents"])
queries = read_jsonl(I["queries"])
texts = [d["text"] for d in docs]
fmt = I.get("model_format", "atelier")
# the base loaded once; each layer tried is the same weights pooled at a different depth
base = make_embedder(fmt, I["base_model"], I.get("tokenizer"), device, layer=-1, max_length=int(P["max_length"]))
base.eval()
n_blocks = count_blocks(base)
ks = sorted(int(k) for k in P["k_values"])

layers = sorted({(n_blocks if int(l) < 0 else min(int(l), n_blocks)) for l in P["layers"]})
if bool(P["sweep_all_layers"]):
    layers = list(range(0, n_blocks + 1, max(1, n_blocks // 8)))
    if n_blocks not in layers:
        layers.append(n_blocks)

rows = []
for li, layer in enumerate(layers):
    emb = base
    emb.layer = layer if layer < n_blocks else -1
    D = emb.encode(texts, batch_size=256, progress=lambda d, t: progress(5 + 85 * (li + 0.7 * d / t) / len(layers), f"layer {layer}: documents {d}/{t}"))
    Q = emb.encode([q["query"] for q in queries], batch_size=256)
    index = Index(D, [{"doc_id": i} for i in range(len(texts))])
    results = index.search(Q, max(ks))
    row = {"layer": layer}
    for k in ks:
        row[f"recall@{k}"] = recall_at_k(results, "doc_id", [q["gold_id"] for q in queries], k)
    rows.append(row)
    progress(5 + 85 * (li + 1) / len(layers), f"layer {layer}: recall@5 {row.get('recall@5', 0):.1%}")

best = max(rows, key=lambda r: r.get(f"recall@{ks[min(1, len(ks) - 1)]}", 0))
(run_dir / "zero_shot.json").write_text(json.dumps({"rows": rows, "best_layer": best["layer"], "blocks": n_blocks}, indent=2))
bm25 = float(I.get("bm25_recall5") or 0)
key5 = "recall@5" if "recall@5" in best else f"recall@{ks[-1]}"

R = Result()
R.metric("zero_shot_recall5", f"Zero-shot {key5}", best[key5], "pct", "kept" if best[key5] > bm25 else "dup", help=f"keyword search gets {bm25:.1%}")
R.metric("best_layer", "Best layer to pool from", best["layer"], "int", "sky", help=f"of {n_blocks}")
R.metric("dim", "Embedding dimension", width(base), "int", "hold", help=f"{I.get('model_label', 'base model')} ({fmt})")
R.metric("gap", "Against keyword search", best[key5] - bm25, "pct", "dup" if best[key5] < bm25 else "kept")
R.chart("layers", "Recall by the layer you pool from", [dict(r, bm25=bm25) for r in rows], "layer", [{"key": f"recall@{k}", "label": f"recall@{k}"} for k in ks] + [{"key": "bm25", "label": "BM25 recall@5", "color": "hold"}], "line", y_domain=[0, 1], note="The last layer is rarely the best. A model trained to predict the next token spends its final layers on that prediction, not on a summary of the text.")
R.chart("compare", "Against keyword search", [{"method": "BM25", "recall": bm25}, {"method": f"zero-shot, layer {best['layer']}", "recall": best[key5]}], "method", [{"key": "recall", "label": key5, "color": "kept"}], "bar", y_domain=[0, 1])
R.table("rows", "Every layer", [{"key": "layer", "label": "Layer"}] + [{"key": f"recall@{k}", "label": f"recall@{k}", "fmt": "pct"} for k in ks], rows)
R.note("An untrained embedder usually loses to keyword search here, and that is the honest starting point. The next step closes the gap by telling the model, explicitly, which queries and documents belong together.")
R.artifact(run_dir / "zero_shot.json", "zero_shot.json")
R.output("zero_shot", str(run_dir / "zero_shot.json")).output("best_layer", best["layer"]).output("zero_shot_recall5", best[key5])
for k in ("documents", "queries", "pairs", "library", "base_model", "tokenizer", "lang", "bm25_recall5", "model_format", "adapter", "model_label", "system", "data_source", "data_label"):
    if k in I:
        R.output(k, I[k])
R.save()
