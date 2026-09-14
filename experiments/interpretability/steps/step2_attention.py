"""Step 2 — the heads that copy, and when they appeared."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import attention_pattern, induction_scores  # noqa: E402

parse_args()
P = params({"seq_len": 48, "n_sequences": 16, "checkpoints": [], "show_head": ""})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
targets = [I["model"]] + [c for c in P["checkpoints"] if c != I["model"]]

all_scores, summaries = {}, []
for ti, name in enumerate(targets):
    try:
        mp = hf.model_path(name)
    except Exception as exc:
        summaries.append({"model": name, "error": str(exc)[:100], "top_score": 0, "heads_above": 0})
        continue
    tok = hf.load_tokenizer(mp)
    model = hf.load_model(mp, dtype="fp32")
    model.eval()
    scores = induction_scores(model, tok, int(P["seq_len"]), int(P["n_sequences"]), progress=lambda d, t: progress(5 + 80 * (ti + d / t) / len(targets), f"{name}: sequence {d}/{t}"))
    all_scores[name] = scores
    flat = scores.flatten()
    uniform = 1.0 / (2 * int(P["seq_len"]))
    summaries.append({"model": name, "top_score": float(flat.max()), "mean_score": float(flat.mean()), "heads_above": int((flat > 10 * uniform).sum()), "heads": int(scores.shape[0] * scores.shape[1]), "uniform": uniform})
    if name == targets[0]:
        base_scores, base_tok, base_model = scores, tok, model
    else:
        del model
        torch.cuda.empty_cache()

scores = all_scores[targets[0]]
n_layers, n_heads = scores.shape
ranked = sorted(((l, h, float(scores[l, h])) for l in range(n_layers) for h in range(n_heads)), key=lambda r: -r[2])
top = ranked[:12]
by_layer = [{"layer": l, "max": float(scores[l].max()), "mean": float(scores[l].mean())} for l in range(n_layers)]
heat = [dict({"layer": l}, **{f"h{h}": float(scores[l, h]) for h in range(n_heads)}) for l in range(n_layers)]

if P["show_head"].strip():
    L, Hd = (int(x) for x in P["show_head"].split("."))
else:
    L, Hd = top[0][0], top[0][1]
probe_text = hf.dataset_split("pretrain", "wikitext-103", "validation", limit=40)
text = next((d["text"] for d in probe_text if len(d.get("text") or "") > 200), "The cat sat on the mat. The cat sat on the")
att, tokens = attention_pattern(base_model, base_tok, text, L, Hd)
window = min(len(tokens), 28)
pattern_rows = [{"query": tokens[i][:12], **{f"k{j}": float(att[i, j]) for j in range(window)}} for i in range(window)]
(run_dir / "attention.json").write_text(json.dumps({"summaries": summaries, "top": top, "shown": [L, Hd]}, indent=2))

R = Result()
R.metric("top_induction", f"Strongest head: layer {top[0][0]}, head {top[0][1]}", top[0][2], "num", "kept", help=f"uniform attention would give {summaries[0]['uniform']:.4f}")
R.metric("heads_above", "Heads well above chance", summaries[0]["heads_above"], "int", "sky", help=f"of {summaries[0]['heads']}")
R.metric("layers", "Layers", n_layers, "int", "hold")
R.metric("shown", "Pattern shown", f"{L}.{Hd}", "text", "raw")
R.chart("heat", "Induction score by layer and head", heat, "layer", [{"key": f"h{h}", "label": f"head {h}"} for h in range(n_heads)], "bar", note="Induction heads cluster in the second half of the network, and there are usually only a handful of strong ones however large the model is.")
R.chart("by_layer", "Strongest head in each layer", by_layer, "layer", [{"key": "max", "label": "Best head", "color": "kept"}, {"key": "mean", "label": "Layer mean", "color": "hold"}], "line")
if len(summaries) > 1:
    R.chart("checkpoints", "Induction across training", [{"model": s["model"].replace("pythia-160m-", "").replace("pythia-160m", "final"), "top_score": s["top_score"], "heads_above": s["heads_above"]} for s in summaries], "model", [{"key": "top_score", "label": "Strongest head", "color": "kept"}, {"key": "heads_above", "label": "Heads above chance", "color": "sky", "axis": "right"}], "bar", note="These are the same model at different points in one training run. Induction appears abruptly rather than gradually, at the point where the loss curve has its well-known bump.")
R.table("top", "The strongest heads", [{"key": "layer", "label": "Layer"}, {"key": "head", "label": "Head"}, {"key": "score", "label": "Induction score", "fmt": "num"}], [{"layer": l, "head": h, "score": s} for l, h, s in top])
R.table("pattern", f"What head {L}.{Hd} attends to", [{"key": "query", "label": "From"}] + [{"key": f"k{j}", "label": tokens[j][:8], "fmt": "num"} for j in range(window)], pattern_rows, note="Rows are query positions, columns are keys. A previous-token head is a line just below the diagonal; an induction head puts its weight far to the left, on the token after the earlier copy.")
R.artifact(run_dir / "attention.json", "attention.json")
R.output("attention", str(run_dir / "attention.json")).output("top_head", f"{top[0][0]}.{top[0][1]}").output("top_induction", top[0][2])
for k in ("model", "probe", "layers", "probe_accuracy"):
    if k in I:
        R.output(k, I[k])
R.save()
