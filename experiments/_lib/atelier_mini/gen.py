"""Batched sampling. Small models are cheap to sample, so evaluation can afford
many completions per prompt — which is what pass@k and GRPO both need."""
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .tok import EOS, PAD


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 160, temperature: float = 0.0, top_k: int = 0, num_samples: int = 1, batch_size: int = 64, system: Optional[str] = None, stop: Optional[str] = None, progress: Optional[Callable[[int, int], None]] = None) -> list[list[str]]:
    """Returns num_samples completions for each prompt (new text only)."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: list[list[str]] = []
    expanded = [p for p in prompts for _ in range(num_samples)]
    done = 0
    for start in range(0, len(expanded), batch_size):
        chunk = expanded[start : start + batch_size]
        encoded = [tok.encode((f"{system}\n{p}\n" if system else f"{p}\n"), bos=True) for p in chunk]
        width = max(len(e) for e in encoded)
        # left padding keeps every sequence's last token at the same index
        idx = torch.full((len(chunk), width), PAD, dtype=torch.long, device=device)
        for i, e in enumerate(encoded):
            idx[i, width - len(e) :] = torch.tensor(e, device=device)
        finished = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        produced = [[] for _ in chunk]
        for _ in range(max_new_tokens):
            window = idx[:, -model.config.block_size :]
            logits, _ = model(window)
            logits = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            nxt[finished] = PAD
            idx = torch.cat([idx, nxt], dim=1)
            for i, t in enumerate(nxt.squeeze(1).tolist()):
                if not finished[i] and t != PAD:
                    produced[i].append(t)
            finished |= nxt.squeeze(1) == EOS
            if bool(finished.all()):
                break
        for i in range(len(chunk)):
            text = tok.decode(produced[i])
            if stop and stop in text:
                text = text.split(stop)[0] + stop
            out.append(text)
        done += len(chunk)
        if progress:
            progress(min(done, len(expanded)), len(expanded))
    if was_training:
        model.train()
    return [out[i * num_samples : (i + 1) * num_samples] for i in range(len(prompts))]


@torch.no_grad()
def sequence_logprobs(model, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token) over the masked positions — the policy's score for a completion."""
    logits, _ = model(idx[:, :-1])
    logp = F.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask[:, 1:]).sum(dim=1)


@torch.no_grad()
def generate_cached(model, tok, prompts: list[str], max_new_tokens: int = 160, temperature: float = 0.0, top_k: int = 0, system: Optional[str] = None, draft=None, draft_lookahead: int = 4):
    """Generation with a key/value cache, and optionally speculative decoding.

    Without a cache every new token re-reads the whole prefix, so cost grows with
    the square of the length. With one, each step attends to stored keys and values
    and costs the same as the first. `draft` turns on speculation: a smaller model
    proposes `draft_lookahead` tokens, this model checks them in a single pass, and
    the run of tokens it agrees with is kept.

    Returns (texts, stats). Implemented straightforwardly for reading; the caching
    lives in model.forward_cached.
    """
    device = next(model.parameters()).device
    outputs, stats = [], {"accepted": 0, "proposed": 0, "forward_passes": 0}
    for prompt in prompts:
        ids = torch.tensor([tok.encode((f"{system}\n{prompt}\n" if system else f"{prompt}\n"), bos=True)], device=device)
        cache = None
        produced: list[int] = []
        while len(produced) < max_new_tokens:
            if draft is not None:
                proposal, _ = generate_cached(draft, tok, [tok.decode(ids[0].tolist())], draft_lookahead, temperature, top_k, None)
                extra = tok.encode(proposal[0])[:draft_lookahead]
                if extra:
                    candidate = torch.cat([ids, torch.tensor([extra], device=device)], dim=1)
                    logits, cache = model.forward_cached(candidate, None)
                    stats["forward_passes"] += 1
                    stats["proposed"] += len(extra)
                    check = logits[0, -len(extra) - 1 : -1].argmax(-1).tolist()
                    agreed = 0
                    for proposed_tok, own in zip(extra, check):
                        if proposed_tok != own:
                            break
                        agreed += 1
                    stats["accepted"] += agreed
                    keep = extra[:agreed] + [check[agreed]] if agreed < len(extra) else extra
                    produced += keep
                    ids = torch.cat([ids, torch.tensor([keep], device=device)], dim=1)
                    cache = None
                    if EOS in keep:
                        break
                    continue
            logits, cache = model.forward_cached(ids if cache is None else ids[:, -1:], cache)
            stats["forward_passes"] += 1
            step = logits[:, -1, :]
            if temperature <= 0:
                nxt = step.argmax(-1, keepdim=True)
            else:
                step = step / max(temperature, 1e-5)
                if top_k:
                    v, _ = torch.topk(step, min(top_k, step.size(-1)))
                    step[step < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(step, dim=-1), 1)
            t = int(nxt)
            produced.append(t)
            ids = torch.cat([ids, nxt], dim=1)
            if t == EOS:
                break
        outputs.append(tok.decode(produced))
    return outputs, stats
