"""Step 1 — the grid, and what it costs before you spend it."""
import json
import os
import sys
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args
from atelier_mini.model import MiniConfig, estimate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_grid import GRID  # noqa: E402

parse_args()
P = params({"sizes": ["n1", "n2", "n3", "n4"], "tokens_per_param": 20.0, "block_size": 256, "batch_size": 48})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("pack_run_run")
if not ref:
    raise SystemExit("Choose a Pretraining step 1 run — every point trains on the same token stream.")
o = ref["outputs"]
vocab = int(o["vocab_size"])
available = int(o["train_tokens"])
A6000 = 155 * 0.35 * 1e12
block, batch = int(P["block_size"]), int(P["batch_size"])
tpp = float(P["tokens_per_param"])

points = []
for key in P["sizes"]:
    L, H, D, label = GRID[key]
    cfg = MiniConfig(vocab_size=vocab, n_layer=L, n_head=H, n_embd=D, block_size=block)
    est = estimate(cfg)
    tokens = int(tpp * est["total"])
    iters = max(50, tokens // (batch * block))
    seconds = est["flops_per_token"] * tokens / A6000
    points.append({"key": key, "label": label, "params": est["total"], "non_embedding": est["non_embedding"], "tokens": tokens, "iters": iters, "hours": seconds / 3600, "config": cfg.to_dict()})

total_hours = sum(p["hours"] for p in points)
(run_dir / "plan.json").write_text(json.dumps({"points": points, "batch_size": batch, "block_size": block, "tokens_per_param": tpp, "vocab_size": vocab}, indent=2))

R = Result()
R.metric("points", "Points in the grid", len(points), "int", "kept")
R.metric("total_hours", "Estimated total time", total_hours * 3600 * 1000, "ms", "raw", help="one A6000, in sequence")
R.metric("span", "Size range", f"{points[0]['params'] / 1e6:.1f}M – {points[-1]['params'] / 1e6:.1f}M", "text", "sky", help="a fit needs at least a factor of ten between the ends to be worth trusting")
R.metric("tokens_needed", "Tokens the grid will consume", sum(p["tokens"] for p in points), "int", "hold", help=f"your stream holds {available:,}; points needing more will repeat data")
R.chart("cost", "Cost per point", points, "label", [{"key": "params", "label": "Parameters", "color": "kept"}, {"key": "hours", "label": "Hours", "color": "raw", "axis": "right"}], "bar")
R.chart("tokens", "Token budget per point", points, "label", [{"key": "tokens", "label": "Tokens", "color": "hold"}], "bar", note=f"At {tpp:.0f} tokens per parameter. Lower it to fit the lesson; the models will be starved, and the fit will show it.")
R.table("grid", "The grid", [{"key": "label", "label": "Shape"}, {"key": "params", "label": "Parameters", "fmt": "int"}, {"key": "non_embedding", "label": "Non-embedding", "fmt": "int"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "iters", "label": "Steps", "fmt": "int"}, {"key": "hours", "label": "Hours", "fmt": "num"}], points)
if sum(p["tokens"] for p in points) > available:
    R.note(f"The grid wants more tokens than your stream holds ({available:,}). Either pack more documents, or accept that the larger points will see the same data more than once — which raises their loss and bends the curve.")
R.artifact(run_dir / "plan.json", "plan.json")
# the data is whatever the Pretraining run packed: generated from the world, or a
# prepared corpus and tokenizer chosen there
source_note = f"Token stream from Pretraining run #{ref['id']}: documents {o.get('data_label', 'generated from the world')}, tokenizer {o.get('tokenizer_label', 'from a Tokenizer run')}."
R.note(source_note)
R.output("plan", str(run_dir / "plan.json")).output("train_bin", o["train_bin"]).output("val_bin", o["val_bin"]).output("meta", o["meta"]).output("vocab_size", vocab).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("data_label", o.get("data_label", "generated from the world"))
R.save()
