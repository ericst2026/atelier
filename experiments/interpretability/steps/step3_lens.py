"""Step 3 — decode every layer as though the model stopped there."""
import json
import os
import sys
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_interp import logit_lens  # noqa: E402

parse_args()
P = params({"prompts": "", "n_auto": 60, "top_k": 5, "apply_norm": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
mp = hf.model_path(I["model"])
tok = hf.load_tokenizer(mp)
model = hf.load_model(mp, dtype="fp32")
model.eval()

prompts = [p.strip() for p in str(P["prompts"]).splitlines() if p.strip()]
auto = []
if int(P["n_auto"]) > 0:
    for d in hf.dataset_split("pretrain", "wikitext-103", "validation", limit=int(P["n_auto"]) * 8):
        text = " ".join((d.get("text") or "").split())
        if len(text) > 120:
            auto.append(text[:300])
        if len(auto) >= int(P["n_auto"]):
            break

traces, decisions, entropies = [], [], []
for i, prompt in enumerate(prompts + auto):
    rows = logit_lens(model, tok, prompt, int(P["top_k"]), bool(P["apply_norm"]))
    decided = next((r["layer"] for r in rows if r["final_rank"] == 0), len(rows) - 1)
    decisions.append(decided / max(len(rows) - 1, 1))
    entropies.append([r["entropy"] for r in rows])
    if i < len(prompts) or len(traces) < 3:
        traces.append({"prompt": prompt, "rows": rows})
    if i % 10 == 0:
        progress(5 + 90 * i / max(len(prompts) + len(auto), 1), f"{i}/{len(prompts) + len(auto)}")

n_layers = len(traces[0]["rows"])
mean_entropy = [{"layer": l, "entropy": sum(e[l] for e in entropies) / len(entropies)} for l in range(n_layers)]
mean_rank = [{"layer": l, "rank": sum(t["rows"][l]["final_rank"] for t in traces) / len(traces), "prob": sum(t["rows"][l]["final_prob"] for t in traces) / len(traces)} for l in range(n_layers)]
decision_layer = sum(decisions) / len(decisions)
(run_dir / "lens.json").write_text(json.dumps({"traces": traces[:6], "decision_fraction": decision_layer}, indent=2, ensure_ascii=False))

R = Result()
R.metric("decision_layer", "Where the final token first leads", decision_layer, "pct", "kept", help="as a fraction of depth, averaged over prompts")
R.metric("layers", "Layers", n_layers - 1, "int", "hold", help=I["model"])
R.metric("final_entropy", "Entropy at the last layer", mean_entropy[-1]["entropy"], "num", "sky", help=f"at the embedding: {mean_entropy[0]['entropy']:.2f}")
R.metric("prompts", "Prompts traced", len(prompts) + len(auto), "int", "raw")
R.chart("entropy", "How uncertain each layer is", mean_entropy, "layer", [{"key": "entropy", "label": "Entropy", "color": "hold"}], "line", note="High and flat early, falling sharply in the last third. The model spends most of its depth deciding what kind of thing comes next, and only the end deciding which.")
R.chart("rank", "The final token's rank, layer by layer", mean_rank, "layer", [{"key": "rank", "label": "Rank of the eventual prediction", "color": "dup"}, {"key": "prob", "label": "Its probability", "color": "kept", "axis": "right"}], "line", y_log=True)
R.chart("decisions", "Where different prompts settle", hist(decisions, bins=12, lo=0, hi=1), "bin", [{"key": "count", "label": "Prompts", "color": "sky"}], "bar", note="Prompts with an obvious continuation settle early; ambiguous ones stay open until the last layers.")
for t in traces[:3]:
    R.table(f"trace_{abs(hash(t['prompt'])) % 10000}", f"“{t['prompt'][:60]}”", [{"key": "layer", "label": "Layer"}, {"key": "top", "label": "Most likely tokens"}, {"key": "prob", "label": "Top probability", "fmt": "pct"}, {"key": "final_rank", "label": "Rank of the final answer", "fmt": "int"}],
            [{"layer": r["layer"], "top": " · ".join(repr(x)[1:-1] for x in r["tokens"]), "prob": r["probs"][0], "final_rank": r["final_rank"]} for r in t["rows"]])
R.note("The logit lens assumes every layer writes into the same space as the output embedding. That is roughly true for models with a residual stream and untied norms, and less true the further you get from GPT-2's design — read the early layers with that in mind.")
R.artifact(run_dir / "lens.json", "lens.json")
R.output("lens", str(run_dir / "lens.json")).output("decision_layer", decision_layer)
for k in ("model", "probe", "attention", "layers", "top_head", "probe_accuracy"):
    if k in I:
        R.output(k, I[k])
R.save()
