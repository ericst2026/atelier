"""Step 2 — what the frozen reference already thinks of each pair."""
import os
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.dpo import _batch, sequence_logprob
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer

parse_args()
P = params({"probe": 400})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
pairs = read_jsonl(I["pairs"], limit=int(P["probe"]))
reference, ck = MiniLM.load(I["policy"], device)
reference.eval()
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or ck.get("system")
block = reference.config.block_size

margins, chosen_lp, rejected_lp = [], [], []
with torch.no_grad():
    for i in range(0, len(pairs), 8):
        chunk = pairs[i : i + 8]
        c_idx, c_mask = _batch(chunk, tok, block, device, system, "chosen")
        r_idx, r_mask = _batch(chunk, tok, block, device, system, "rejected")
        lc, nc = sequence_logprob(reference, c_idx, c_mask)
        lr, nr = sequence_logprob(reference, r_idx, r_mask)
        margins += (lc - lr).tolist()
        chosen_lp += (lc / nc.clamp(min=1)).tolist()
        rejected_lp += (lr / nr.clamp(min=1)).tolist()
        progress(5 + 90 * i / len(pairs), f"{i}/{len(pairs)} pairs")

already = sum(1 for m in margins if m > 0) / len(margins)
settled = sum(1 for m in margins if abs(m) > 20) / len(margins)
R = Result()
R.metric("already_right", "Pairs the reference already prefers", already, "pct", "hold", help="DPO only has to move the rest")
R.metric("settled", "Pairs the reference is already sure about", settled, "pct", "dup", help="a very large margin either way contributes almost nothing to the gradient")
R.metric("mean_margin", "Mean log-probability margin", sum(margins) / len(margins), "num", "sky")
R.metric("chosen_logp", "Chosen, per token", sum(chosen_lp) / len(chosen_lp), "num", "kept", help=f"rejected: {sum(rejected_lp) / len(rejected_lp):.3f}")
R.chart("margins", "Reference margin per pair", hist(margins, bins=25), "bin", [{"key": "count", "label": "Pairs", "color": "sky"}], "bar", note="Mass to the right of zero is where the reference already agrees with your labels. The useful pairs sit near the middle.")
R.chart("logp", "Per-token log-probability", [{"kind": "chosen", "value": sum(chosen_lp) / len(chosen_lp)}, {"kind": "rejected", "value": sum(rejected_lp) / len(rejected_lp)}], "kind", [{"key": "value", "label": "Log-probability", "color": "kept"}], "bar")
R.output("reference", I["policy"])
for k in ("pairs", "pairs_val", "policy", "tokenizer", "lang", "system", "sft_accuracy"):
    if k in I:
        R.output(k, I[k])
R.save()
