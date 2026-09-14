"""Direct preference optimisation.

No reward model and no sampling: for each (prompt, chosen, rejected) triple, raise
the policy's log-probability of the chosen answer relative to the rejected one,
measured against a frozen reference so the model cannot simply become more
confident about everything."""
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .tok import PAD


def _encode(tok, prompt: str, answer: str, block_size: int, device, system: Optional[str]):
    p = tok.encode((f"{system}\n{prompt}\n" if system else f"{prompt}\n"), bos=True)
    a = tok.encode(answer, eos=True)
    ids = (p + a)[:block_size]
    mask = ([0] * len(p) + [1] * len(a))[:block_size]
    return ids, mask


def _batch(rows, tok, block_size, device, system, key):
    seqs, masks = [], []
    for r in rows:
        ids, m = _encode(tok, r["prompt"], r[key], block_size, device, system)
        seqs.append(ids)
        masks.append(m)
    width = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), width), PAD, dtype=torch.long)
    msk = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        idx[i, : len(s)] = torch.tensor(s)
        msk[i, : len(m)] = torch.tensor(m)
    return idx.to(device), msk.to(device)


def sequence_logprob(model, idx, mask):
    logits, _ = model(idx[:, :-1])
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
    m = mask[:, 1:].float()
    return (tok_lp * m).sum(dim=1), m.sum(dim=1)


def dpo(model, reference, tok, pairs: list[dict], out_dir: Path, val_pairs: Optional[list[dict]] = None, beta: float = 0.1, epochs: float = 1.0, batch_size: int = 8, lr: float = 5e-6, warmup: int = 20, block_size: Optional[int] = None, system: Optional[str] = None, label_smoothing: float = 0.0, device: Optional[str] = None, eval_every: int = 50, on_log: Optional[Callable[[dict], None]] = None) -> dict:
    """pairs: [{"prompt", "chosen", "rejected"}]. Returns history with accuracy and margins."""
    import random

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from .train import cosine_lr

    block_size = block_size or model.config.block_size
    opt = model.optimizers(0.0, lr)
    steps = max(1, int(len(pairs) * epochs / batch_size))
    rng = random.Random(1)
    order = list(pairs)
    history, t0 = [], time.time()
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    def evaluate(rows) -> dict:
        model.eval()
        acc, margin, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(rows), 256), batch_size):
                chunk = rows[i : i + batch_size]
                stats = _step(chunk, train=False)
                acc += stats["accuracy"] * len(chunk)
                margin += stats["margin"] * len(chunk)
                n += len(chunk)
        model.train()
        return {"eval_accuracy": acc / max(n, 1), "eval_margin": margin / max(n, 1)}

    def _step(chunk, train: bool) -> dict:
        c_idx, c_mask = _batch(chunk, tok, block_size, device, system, "chosen")
        r_idx, r_mask = _batch(chunk, tok, block_size, device, system, "rejected")
        pi_c, n_c = sequence_logprob(model, c_idx, c_mask)
        pi_r, n_r = sequence_logprob(model, r_idx, r_mask)
        with torch.no_grad():
            ref_c, _ = sequence_logprob(reference, c_idx, c_mask)
            ref_r, _ = sequence_logprob(reference, r_idx, r_mask)
        logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        loss = -(F.logsigmoid(logits) * (1 - label_smoothing) + F.logsigmoid(-logits) * label_smoothing).mean()
        if train:
            loss.backward()
        return {
            "loss": float(loss),
            "accuracy": float((logits > 0).float().mean()),
            "margin": float(((pi_c - ref_c) - (pi_r - ref_r)).mean()),
            "chosen_logp": float((pi_c / n_c.clamp(min=1)).mean()),
            "rejected_logp": float((pi_r / n_r.clamp(min=1)).mean()),
        }

    model.train()
    cursor = 0
    for it in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, steps, lr, warmup)
        if it % eval_every == 0 or it == steps:
            row = {"step": it, "lr": opt.param_groups[0]["lr"], "elapsed": time.time() - t0}
            if val_pairs:
                row.update(evaluate(val_pairs))
            history.append(row)
            if on_log:
                on_log(row)
            if it == steps:
                break
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        chunk = order[cursor : cursor + batch_size]
        cursor += batch_size
        stats = _step(chunk, train=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if on_log and it % 5 == 0:
            on_log({"step": it, **stats})
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.pt", {"dpo": True, "beta": beta, "steps": steps})
    return {"history": history, "elapsed_sec": time.time() - t0, "checkpoint": str(out_dir / "model.pt"), "steps": steps}
