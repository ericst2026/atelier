"""Step 2 — the heads that copy, and whether they were always there."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.model import MiniLM
from atelier_mini.tok import load_tokenizer
from atelier_world import World
from atelier_world.prepared import choose_model, read_documents

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import induction_scores  # noqa: E402

parse_args()
P = params({"seq_len": 48, "n_sequences": 16, "compare_source": "generated", "compare_a": None, "compare_b": None, "compare_a_material": None, "compare_b_material": None, "text_source": "generated", "text_material": None, "show_head": ""})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = load_tokenizer(I["tokenizer"])
targets = [("this model", I["model"])]
prepared_compare = P["compare_source"] == "prepared"
for key in ("compare_a", "compare_b"):
    ref = I.get(f"{key}_run")
    if (prepared_compare and P.get(f"{key}_material")) or (not prepared_compare and ref):
        # induction scores come from the Atelier model's attention maps, so HuggingFace is refused
        c = choose_model(P, I, run_key=key, run_model_key="sft_model" if ref and ref["outputs"].get("sft_model") else "model", hint="", source_key="compare_source", material_key=f"{key}_material", formats=("atelier",))
        label = f"run {ref['id']} · {c['outputs'].get('tokens_seen', 0):,} tokens" if c["source"] == "generated" else c["label"]
        targets.append((label, c["model"]))

summaries, all_scores = [], {}
for ti, (label, path) in enumerate(targets):
    model, _ = MiniLM.load(path, device)
    model.eval()
    scores = induction_scores(model, tok, int(P["seq_len"]), int(P["n_sequences"]), progress=lambda d, t: progress(5 + 75 * (ti + d / t) / len(targets), f"{label}: sequence {d}/{t}"))
    all_scores[label] = scores
    uniform = 1.0 / (2 * int(P["seq_len"]))
    flat = scores.flatten()
    summaries.append({"model": label, "top_score": float(flat.max()), "mean_score": float(flat.mean()), "heads_above": int((flat > 10 * uniform).sum()), "heads": int(flat.size), "uniform": uniform, "params": model.num_params()})
    if ti == 0:
        base_model = model
    else:
        del model
        torch.cuda.empty_cache()

scores = all_scores[targets[0][0]]
n_layers, n_heads = scores.shape
ranked = sorted(((l, h, float(scores[l, h])) for l in range(n_layers) for h in range(n_heads)), key=lambda r: -r[2])
top = ranked[:12]
heat = [dict({"layer": l}, **{f"h{h}": float(scores[l, h]) for h in range(n_heads)}) for l in range(n_layers)]
by_layer = [{"layer": l, "max": float(scores[l].max()), "mean": float(scores[l].mean())} for l in range(n_layers)]

L, Hd = (int(x) for x in P["show_head"].split(".")) if P["show_head"].strip() else (top[0][0], top[0][1])
if P["text_source"] == "prepared":
    if not P.get("text_material"):
        raise SystemExit("Choose a prepared text dataset, or switch the text back to generated.")
    text = read_documents(P["text_material"], limit=1)[0]["text"]
else:
    world = World(lang=I.get("lang", "en"), seed=3)
    text = world.documents(1).__next__()["text"]
ids = torch.tensor([tok.encode(text, bos=True)[:28]], device=device)
att = base_model.attention_maps(ids)[L, 0, Hd].float().cpu().numpy()
pieces = [tok.decode([i]) or "·" for i in ids[0].tolist()]
window = len(pieces)
pattern_rows = [{"query": pieces[i][:10], **{f"k{j}": float(att[i, j]) for j in range(window)}} for i in range(window)]
(run_dir / "attention.json").write_text(json.dumps({"summaries": summaries, "top": top, "shown": [L, Hd]}, indent=2))

R = Result()
R.metric("top_induction", f"Strongest head: layer {top[0][0]}, head {top[0][1]}", top[0][2], "num", "kept", help=f"uniform attention would give {summaries[0]['uniform']:.4f}")
R.metric("heads_above", "Heads well above chance", summaries[0]["heads_above"], "int", "sky", help=f"of {summaries[0]['heads']}")
R.metric("layers", "Layers × heads", f"{n_layers} × {n_heads}", "text", "hold")
R.metric("shown", "Pattern shown", f"{L}.{Hd}", "text", "raw")
R.chart("heat", "Induction score by layer and head", heat, "layer", [{"key": f"h{h}", "label": f"head {h}"} for h in range(n_heads)], "bar", note="Even in a small model these cluster: a handful of heads do most of the copying, usually in the second half.")
R.chart("by_layer", "Strongest head in each layer", by_layer, "layer", [{"key": "max", "label": "Best head", "color": "kept"}, {"key": "mean", "label": "Layer mean", "color": "hold"}], "line")
if len(summaries) > 1:
    R.chart("models", "Across your runs", [{"model": s["model"], "top_score": s["top_score"], "heads_above": s["heads_above"]} for s in summaries], "model", [{"key": "top_score", "label": "Strongest head", "color": "kept"}, {"key": "heads_above", "label": "Heads above chance", "color": "sky", "axis": "right"}], "bar", note="Compare pretraining runs of different lengths and the heads appear rather than grow — the same abrupt arrival the published checkpoints are famous for, in a model you trained yourself.")
R.table("top", "The strongest heads", [{"key": "layer", "label": "Layer"}, {"key": "head", "label": "Head"}, {"key": "score", "label": "Induction score", "fmt": "num"}], [{"layer": l, "head": h, "score": s} for l, h, s in top])
R.table("pattern", f"What head {L}.{Hd} attends to", [{"key": "query", "label": "From"}] + [{"key": f"k{j}", "label": pieces[j][:6], "fmt": "num"} for j in range(window)], pattern_rows, note="Rows are query positions, columns are keys. A previous-token head is a line just below the diagonal; an induction head puts weight far to the left.")
R.artifact(run_dir / "attention.json", "attention.json")
R.output("attention", str(run_dir / "attention.json")).output("top_head", f"{top[0][0]}.{top[0][1]}").output("top_induction", top[0][2])
for k in ("model", "tokenizer", "lang", "system", "probe", "layers", "probe_accuracy", "model_format", "adapter", "model_label", "data_source", "data_label", "questions"):
    if k in I:
        R.output(k, I[k])
R.save()
