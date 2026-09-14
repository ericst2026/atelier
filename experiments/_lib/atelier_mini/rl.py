"""GRPO in about a hundred lines.

For each prompt, sample a group of completions, score them with a reward
function, and use the group's own mean as the baseline: completions above it are
made more likely, those below less. No value network, no critic — the group is
the baseline. A KL term to the starting policy keeps the model from drifting into
gibberish that happens to score well."""
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .gen import generate
from .tok import EOS, PAD


def _encode_pairs(tok, prompts: list[str], completions: list[str], block_size: int, device, system: Optional[str]):
    seqs, masks = [], []
    for p, c in zip(prompts, completions):
        p_ids = tok.encode((f"{system}\n{p}\n" if system else f"{p}\n"), bos=True)
        c_ids = tok.encode(c, eos=True)
        ids = (p_ids + c_ids)[:block_size]
        m = ([0] * len(p_ids) + [1] * len(c_ids))[:block_size]
        seqs.append(ids)
        masks.append(m)
    width = max(len(s) for s in seqs)
    idx = torch.full((len(seqs), width), PAD, dtype=torch.long)
    msk = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        idx[i, : len(s)] = torch.tensor(s)
        msk[i, : len(m)] = torch.tensor(m)
    return idx.to(device), msk.to(device)


def _token_logprobs(model, idx, mask):
    logits, _ = model(idx[:, :-1])
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok_lp, mask[:, 1:].float()


def grpo(
    model,
    tok,
    prompts: list[dict],
    reward_fn: Callable[[dict, str], float],
    out_dir: Path,
    reference=None,
    group_size: int = 8,
    prompts_per_step: int = 8,
    max_new_tokens: int = 160,
    temperature: float = 1.0,
    top_k: int = 0,
    beta: float = 0.02,
    lr: float = 1e-5,
    max_steps: int = 200,
    clip_eps: float = 0.2,
    system: Optional[str] = None,
    device: str = "cuda",
    on_log: Optional[Callable[[dict], None]] = None,
    seed: int = 1,
) -> dict:
    import random

    rng = random.Random(seed)
    opt = model.optimizers(0.0, lr)
    history, t0 = [], time.time()
    block = model.config.block_size
    for step in range(max_steps):
        batch = [rng.choice(prompts) for _ in range(prompts_per_step)]
        texts = [b["prompt"] for b in batch]
        samples = generate(model, tok, texts, max_new_tokens, max(temperature, 0.1), top_k, num_samples=group_size, batch_size=max(8, group_size * 2), system=system)

        flat_prompts, flat_completions, rewards = [], [], []
        for b, group in zip(batch, samples):
            for c in group:
                flat_prompts.append(b["prompt"])
                flat_completions.append(c)
                rewards.append(float(reward_fn(b, c)))
        r = torch.tensor(rewards, dtype=torch.float32, device=device).view(prompts_per_step, group_size)
        # the group is its own baseline: centre, then scale so groups of mixed
        # difficulty contribute comparably
        adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-4)
        adv = adv.view(-1)

        idx, mask = _encode_pairs(tok, flat_prompts, flat_completions, block, device, system)
        with torch.no_grad():
            old_lp, m = _token_logprobs(model, idx, mask)
            ref_lp = _token_logprobs(reference, idx, mask)[0] if reference is not None else None

        model.train()
        new_lp, m = _token_logprobs(model, idx, mask)
        ratio = torch.exp(new_lp - old_lp)
        a = adv.unsqueeze(1)
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a
        policy_loss = -(torch.min(unclipped, clipped) * m).sum() / m.sum().clamp(min=1)
        kl = torch.tensor(0.0, device=device)
        if ref_lp is not None and beta > 0:
            # k3 estimator: non-negative and low variance
            diff = ref_lp - new_lp
            kl = ((torch.exp(diff) - diff - 1) * m).sum() / m.sum().clamp(min=1)
        loss = policy_loss + beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        lengths = m.sum(dim=1)
        row = {
            "step": step,
            "reward": float(r.mean()),
            "reward_std": float(r.std()),
            "solved": float((r > r.min()).float().mean()) if float(r.max()) > float(r.min()) else 0.0,
            "kl": float(kl),
            "completion_length": float(lengths.float().mean()),
            "loss": float(loss),
            "elapsed": time.time() - t0,
        }
        history.append(row)
        if on_log:
            on_log(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.pt", {"rl": True, "steps": max_steps})
    return {"history": history, "elapsed_sec": time.time() - t0, "checkpoint": str(out_dir / "model.pt")}
