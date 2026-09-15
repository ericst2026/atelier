"""Scoring a MiniLM (or a HuggingFace causal LM) on a benchmark.

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


@torch.no_grad()
def option_logprobs_hf(model, tok, context: str, options: list[str], batch_size: int = 16, max_length: int = 2048) -> list[dict[str, float]]:
    """The same three numbers for a HuggingFace causal LM, on the raw text (no chat
    template), the way public harnesses score multiple choice. The context and the
    option are tokenised separately so the option's tokens are exactly its own."""
    device = next(model.parameters()).device
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else (start or 0)
    ctx_ids = tok(context + "\n", add_special_tokens=False)["input_ids"]
    out: list[dict[str, float]] = []

    def score(batch: list[list[int]], n_scored: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(b) for b in batch)
        idx = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        att = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        m = torch.zeros((len(batch), width), dtype=torch.float32, device=device)
        for i, b in enumerate(batch):
            idx[i, : len(b)] = torch.tensor(b, device=device)
            att[i, : len(b)] = 1
            m[i, len(b) - n_scored[i] : len(b)] = 1.0
        logits = model(input_ids=idx, attention_mask=att).logits[:, :-1]
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_lp = logp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
        mm = m[:, 1:]
        return (tok_lp * mm).sum(1), mm.sum(1).clamp(min=1)

    for s in range(0, len(options), batch_size):
        chunk = options[s : s + batch_size]
        seqs, alone, n_opt = [], [], []
        for opt in chunk:
            opt_ids = tok(" " + opt.strip(), add_special_tokens=False)["input_ids"][: max_length - 2]
            keep = max_length - 1 - len(opt_ids)
            # a context too long for the model loses its beginning, never the option
            seqs.append([start] + ctx_ids[-keep:] + opt_ids if keep > 0 else [start] + opt_ids)
            alone.append([start] + opt_ids)
            n_opt.append(len(opt_ids))
        totals, counts = score(seqs, n_opt)
        alone_lp, _ = score(alone, n_opt)
        for i in range(len(chunk)):
            out.append({
                "sum": float(totals[i]),
                "tokens": float(counts[i]),
                "mean": float(totals[i] / counts[i]),
                "alone": float(alone_lp[i]),
                "unconditional": float(totals[i] - alone_lp[i]),
            })
    return out


def score_options(lm, context: str, options: list[str], batch_size: int = 16) -> list[dict[str, float]]:
    """option_logprobs for either kind of model loaded with atelier_world.prepared.load_lm."""
    if lm.format == "hf":
        return option_logprobs_hf(lm.model, lm.tok, context, options, batch_size)
    return option_logprobs(lm.model, lm.tok, context, options, batch_size)


def pick(scores: list[dict[str, float]], method: str) -> int:
    key = {"sum": "sum", "mean": "mean", "unconditional": "unconditional"}[method]
    return max(range(len(scores)), key=lambda i: scores[i][key])


from lib_items import few_shot_prefix, render_item  # noqa: E402,F401  (re-exported)
