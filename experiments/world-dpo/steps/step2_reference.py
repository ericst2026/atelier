"""Step 2 — what the frozen reference already thinks of each pair."""
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, read_jsonl
from atelier_world.prepared import load_lm

parse_args()
P = params({"probe": 400})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
pairs = read_jsonl(I["pairs"], limit=int(P["probe"]))
fmt = I.get("model_format", "atelier")
lm = load_lm({"format": fmt, "model": I["policy"], "tokenizer": I["tokenizer"], "adapter": I.get("adapter"), "system": I.get("system")}, device)
reference, tok = lm.model, lm.tok
reference.eval()
system = I.get("system") or lm.system

if fmt == "hf":
    from atelier_nlp import hf

    def logprobs(chunk, key):
        """Summed log-probability of each answer after its chat prompt, and its token count."""
        seqs, starts = [], []
        for r in chunk:
            p_ids = tok(hf.chat_prompt(tok, r["prompt"], system), add_special_tokens=False)["input_ids"]
            a_ids = tok(r[key] + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
            seqs.append((p_ids + a_ids)[:1024])
            starts.append(min(len(p_ids), 1024))
        width = max(len(s) for s in seqs)
        idx = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.float)
        for i, (s, st) in enumerate(zip(seqs, starts)):
            idx[i, : len(s)] = torch.tensor(s)
            attn[i, : len(s)] = 1
            mask[i, st : len(s)] = 1
        idx, attn, mask = idx.to(device), attn.to(device), mask.to(device)
        logits = reference(input_ids=idx, attention_mask=attn).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
        m = mask[:, 1:]
        return (lp * m).sum(1).cpu(), m.sum(1).cpu()
else:
    from atelier_mini.dpo import _batch, sequence_logprob

    block = reference.config.block_size

    def logprobs(chunk, key):
        c_idx, c_mask = _batch(chunk, tok, block, device, system, key)
        return sequence_logprob(reference, c_idx, c_mask)

margins, chosen_lp, rejected_lp = [], [], []
with torch.no_grad():
    for i in range(0, len(pairs), 8):
        chunk = pairs[i : i + 8]
        lc, nc = logprobs(chunk, "chosen")
        lr, nr = logprobs(chunk, "rejected")
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
for k in ("pairs", "pairs_val", "policy", "tokenizer", "lang", "system", "sft_accuracy", "model_format", "adapter", "model_label", "data_source", "eval_questions"):
    if k in I:
        R.output(k, I[k])
R.save()
