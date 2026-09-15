"""Step 4 — needle at every depth and every length."""
import json
import os
import random
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_needle import needle_prompt, plant  # noqa: E402

parse_args()
P = params({"lengths": ["512", "1024", "2048"], "depths": 10, "per_cell": 8})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=77)
tok = MiniTokenizer.load(I["tokenizer"])
lengths = sorted(int(x) for x in P["lengths"])
depths = [round(i / (int(P["depths"]) - 1), 3) for i in range(int(P["depths"]))] if int(P["depths"]) > 1 else [0.5]
rng = random.Random(101)
passages = json.loads(Path(I["filler_eval"]).read_text(encoding="utf-8")) if I.get("data_source") == "prepared" and I.get("filler_eval") else None
cells = [(L, d) for L in lengths for d in depths]

out = {}
for mi, (name, path) in enumerate((("before", I["model"]), ("after", I["long_model"]))):
    model, ck = MiniLM.load(path, device)
    if name == "after":
        model.config.rope_scale = float(ck.get("rope_scale", I["rope_scale"]))
        model.config.rope_base = float(ck.get("rope_base", I["rope_base"]))
    model.config.block_size = max(lengths)
    model._cos = model._sin = None
    grid = {}
    for ci, (L, d) in enumerate(cells):
        needles = [plant(world, tok, L - 60, d, rng, passages) for _ in range(int(P["per_cell"]))]
        gens = generate(model, tok, [needle_prompt(nd["document"], nd["question"], world.answer_prefix) for nd in needles], 24, 0.0, batch_size=4)
        grid[(L, d)] = sum(1 for g, nd in zip(gens, needles) if nd["answer"] in g[0]) / len(needles)
        progress(5 + 45 * mi + 40 * (ci + 1) / len(cells), f"{name}: {L} tokens, depth {d:.0%} → {grid[(L, d)]:.0%}")
    out[name] = grid
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

b, a = out["before"], out["after"]
overall_a = sum(a.values()) / len(a)
overall_b = sum(b.values()) / len(b)
by_depth = [{"depth": f"{int(d * 100)}%", **{f"{L}": a[(L, d)] for L in lengths}} for d in depths]
by_length = [{"length": L, "before": sum(b[(L, d)] for d in depths) / len(depths), "after": sum(a[(L, d)] for d in depths) / len(depths)} for L in lengths]
longest = lengths[-1]
middle = [d for d in depths if 0.3 <= d <= 0.7]
middle_acc = sum(a[(longest, d)] for d in middle) / max(len(middle), 1)
edge = [d for d in depths if d < 0.2 or d > 0.8]
edge_acc = sum(a[(longest, d)] for d in edge) / max(len(edge), 1)

R = Result()
R.metric("needle_accuracy", "Needle found, overall", overall_a, "pct", "kept", help=f"before extending: {overall_b:.1%}")
R.metric("at_longest", f"At {longest} tokens", sum(a[(longest, d)] for d in depths) / len(depths), "pct", "sky")
R.metric("middle", "In the middle of the document", middle_acc, "pct", "dup" if middle_acc < edge_acc * 0.7 else "hold", help=f"at the edges: {edge_acc:.1%}")
R.metric("val_loss", "Validation loss", I.get("val_loss"), "num", "raw")
R.chart("depths", "Needle found by depth", by_depth, "depth", [{"key": str(L), "label": f"{L} tokens"} for L in lengths], "line", y_domain=[0, 1], note="The usual shape is a U: what is at the start and at the end is found, what is in the middle is not.")
R.chart("lengths", "Before and after, by length", by_length, "length", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1])
R.table("grid", "Every cell", [{"key": "length", "label": "Tokens"}, {"key": "depth", "label": "Depth"}, {"key": "before", "label": "Before", "fmt": "pct"}, {"key": "after", "label": "After", "fmt": "pct"}], [{"length": L, "depth": f"{int(d * 100)}%", "before": b[(L, d)], "after": a[(L, d)]} for L in lengths for d in depths])
R.output("needle_accuracy", overall_a).output("long_model", I["long_model"]).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en")).output("model_format", "atelier").output("data_source", I.get("data_source", "generated"))
R.save()
