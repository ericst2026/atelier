"""The scoring itself. Multiple choice never generates — it compares log-probabilities."""
from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def option_logprobs(model, tok, context: str, options: list[str], batch_size: int = 8) -> list[dict]:
    """For each option: total log-probability given the context, the token count, and
    the option's log-probability on its own (the unconditional baseline)."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(options), batch_size):
        chunk = options[start : start + batch_size]
        texts = [context + " " + o.strip() for o in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        ctx_len = len(tok(context, truncation=True, max_length=1024)["input_ids"])
        logits = model(**enc).logits.float()
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        ids = enc["input_ids"][:, 1:]
        token_lp = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        mask = enc["attention_mask"][:, 1:].clone()
        mask[:, : max(ctx_len - 1, 0)] = 0
        totals = (token_lp * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        # the same options with no context, for the unconditional normalisation
        enc2 = tok([o.strip() for o in chunk], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        logits2 = model(**enc2).logits.float()
        logp2 = F.log_softmax(logits2[:, :-1], dim=-1)
        ids2 = enc2["input_ids"][:, 1:]
        lp2 = logp2.gather(-1, ids2.unsqueeze(-1)).squeeze(-1)
        m2 = enc2["attention_mask"][:, 1:]
        alone = (lp2 * m2).sum(dim=1)
        for i in range(len(chunk)):
            out.append({"sum": float(totals[i]), "tokens": int(counts[i]), "mean": float(totals[i] / counts[i]), "alone": float(alone[i]), "unconditional": float(totals[i] - alone[i])})
    return out


def pick(scores: list[dict], method: str) -> int:
    key = {"sum": "sum", "mean": "mean", "unconditional": "unconditional"}[method]
    return max(range(len(scores)), key=lambda i: scores[i][key])
