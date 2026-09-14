"""Scoring a MiniLM on a generated benchmark.

Multiple choice never generates: every option is scored by its log-probability given
the question, and the highest wins. How you normalise that is the decision that moves
published numbers."""
from typing import Any, Optional

import torch
import torch.nn.functional as F

from atelier_mini.tok import PAD


@torch.no_grad()
def option_logprobs(model, tok, context: str, options: list[str], batch_size: int = 16) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    ctx_ids = tok.encode(context + "\n", bos=True)
    out: list[dict[str, float]] = []
    for start in range(0, len(options), batch_size):
        chunk = options[start : start + batch_size]
        seqs, masks, alone_seqs = [], [], []
        for opt in chunk:
            opt_ids = tok.encode(" " + opt.strip())
            ids = (ctx_ids + opt_ids)[: model.config.block_size]
            mask = ([0] * len(ctx_ids) + [1] * len(opt_ids))[: model.config.block_size]
            seqs.append(ids)
            masks.append(mask)
            alone_seqs.append(tok.encode(" " + opt.strip(), bos=True)[: model.config.block_size])

        def score(batch: list[list[int]], msk: Optional[list[list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
            width = max(len(b) for b in batch)
            idx = torch.full((len(batch), width), PAD, dtype=torch.long, device=device)
            m = torch.zeros((len(batch), width), dtype=torch.float32, device=device)
            for i, b in enumerate(batch):
                idx[i, : len(b)] = torch.tensor(b, device=device)
                if msk is None:
                    m[i, 1 : len(b)] = 1.0
                else:
                    mm = msk[i]
                    m[i, : len(mm)] = torch.tensor(mm, dtype=torch.float32, device=device)
            logits, _ = model(idx[:, :-1])
            logp = F.log_softmax(logits.float(), dim=-1)
            tok_lp = logp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
            mm = m[:, 1:]
            return (tok_lp * mm).sum(1), mm.sum(1).clamp(min=1)

        totals, counts = score(seqs, masks)
        alone, _ = score(alone_seqs, None)
        for i in range(len(chunk)):
            out.append({
                "sum": float(totals[i]),
                "tokens": float(counts[i]),
                "mean": float(totals[i] / counts[i]),
                "alone": float(alone[i]),
                "unconditional": float(totals[i] - alone[i]),
            })
    return out


def pick(scores: list[dict[str, float]], method: str) -> int:
    key = {"sum": "sum", "mean": "mean", "unconditional": "unconditional"}[method]
    return max(range(len(scores)), key=lambda i: scores[i][key])


from lib_items import few_shot_prefix, render_item  # noqa: E402,F401  (re-exported)
