"""Step 3 — decode each layer as though the model stopped there."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.model import MiniLM
from atelier_mini.tok import load_tokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import logit_lens  # noqa: E402

parse_args()
P = params({"prompts": "", "n_auto": 60, "top_k": 5})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = MiniLM.load(I["model"], device)
model.eval()
tok = load_tokenizer(I["tokenizer"])
world = World(lang=I.get("lang", "en"), seed=3)

prompts = [p.strip() for p in str(P["prompts"]).splitlines() if p.strip()]
if not int(P["n_auto"]):
    auto = []
elif I.get("data_source") == "prepared":
    # the prepared questions from step 1, counted from the end so they differ from the first ones probed
    auto = [t["prompt"] for t in read_jsonl(I["questions"])[-int(P["n_auto"]):]]
else:
    auto = [t["prompt"] for t in world.eval_set(int(P["n_auto"]), seed=777_001)]
all_prompts = prompts + auto
if not all_prompts:
    raise SystemExit("Give at least one prompt, or ask for some generated ones.")

traces, decisions, entropies = [], [], []
for i, prompt in enumerate(all_prompts):
    rows = logit_lens(model, tok, prompt, int(P["top_k"]))
    decided = next((r["layer"] for r in rows if r["final_rank"] == 0), len(rows) - 1)
    decisions.append(decided / max(len(rows) - 1, 1))
    entropies.append([r["entropy"] for r in rows])
    if i < max(len(prompts), 3):
        traces.append({"prompt": prompt, "rows": rows})
    if i % 10 == 0:
        progress(5 + 90 * i / len(all_prompts), f"{i}/{len(all_prompts)}")

n_layers = len(traces[0]["rows"])
mean_entropy = [{"layer": l, "entropy": sum(e[l] for e in entropies) / len(entropies)} for l in range(n_layers)]
mean_rank = [{"layer": l, "rank": sum(t["rows"][l]["final_rank"] for t in traces) / len(traces), "prob": sum(t["rows"][l]["final_prob"] for t in traces) / len(traces)} for l in range(n_layers)]
decision = sum(decisions) / len(decisions)
(run_dir / "lens.json").write_text(json.dumps({"traces": traces[:6], "decision_fraction": decision}, indent=2, ensure_ascii=False))

R = Result()
R.metric("decision_layer", "Where the final token first leads", decision, "pct", "kept", help="as a fraction of depth, averaged over prompts")
R.metric("layers", "Layers", n_layers - 1, "int", "hold")
R.metric("final_entropy", "Entropy at the last layer", mean_entropy[-1]["entropy"], "num", "sky", help=f"at the embedding: {mean_entropy[0]['entropy']:.2f}")
R.metric("prompts", "Prompts traced", len(all_prompts), "int", "raw")
R.chart("entropy", "How uncertain each layer is", mean_entropy, "layer", [{"key": "entropy", "label": "Entropy", "color": "hold"}], "line", note="High and flat early, falling in the last third. Most of the depth decides what kind of thing comes next; only the end decides which.")
R.chart("rank", "The final token's rank, layer by layer", mean_rank, "layer", [{"key": "rank", "label": "Rank of the eventual prediction", "color": "dup"}, {"key": "prob", "label": "Its probability", "color": "kept", "axis": "right"}], "line", y_log=True)
R.chart("decisions", "Where different prompts settle", hist(decisions, bins=12, lo=0, hi=1), "bin", [{"key": "count", "label": "Prompts", "color": "sky"}], "bar", note="Prompts with an obvious continuation settle early; ambiguous ones stay open to the last layers.")
for t in traces[:3]:
    R.table(f"trace_{abs(hash(t['prompt'])) % 10000}", f"“{t['prompt'][:60]}”", [{"key": "layer", "label": "Layer"}, {"key": "top", "label": "Most likely tokens"}, {"key": "prob", "label": "Top probability", "fmt": "pct"}, {"key": "final_rank", "label": "Rank of the final answer", "fmt": "int"}],
            [{"layer": r["layer"], "top": " · ".join(repr(x)[1:-1] for x in r["tokens"]), "prob": r["probs"][0], "final_rank": r["final_rank"]} for r in t["rows"]])
R.note("The logit lens assumes every layer writes into the space the output embedding reads. Here that is exactly true — the embeddings are tied, so this is the same matrix the last layer uses, which makes the early layers more honest to read than on models where they are not.")
R.artifact(run_dir / "lens.json", "lens.json")
R.output("lens", str(run_dir / "lens.json")).output("decision_layer", decision)
for k in ("model", "tokenizer", "lang", "system", "layers", "probe", "attention", "probe_accuracy", "model_format", "adapter", "model_label", "data_source", "data_label", "questions"):
    if k in I:
        R.output(k, I[k])
R.save()
