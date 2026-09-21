"""The two supervised loops: pretraining on a token stream, and SFT on pairs."""
import math
import time
from pathlib import Path
from typing import Callable, Optional

import torch

from .data import TokenStream, sft_batch


def cosine_lr(it: int, max_iters: int, lr: float, warmup: int, floor: float = 0.1) -> float:
    if it < warmup:
        return lr * (it + 1) / max(1, warmup)
    p = (it - warmup) / max(1, max_iters - warmup)
    return lr * floor + 0.5 * (lr - lr * floor) * (1 + math.cos(math.pi * min(p, 1.0)))


def memory_limit() -> int:
    """Bytes this process may use: the container's limit, else the machine's memory."""
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
            if v.isdigit() and int(v) < 1 << 50:
                return int(v)
        except OSError:
            pass
    try:
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 8 << 30


def micro_batch(model, batch_size: int, block_size: int, device: str) -> int:
    """How many sequences go through the model at once. On a GPU, all of them. On
    a CPU the activations of a whole batch can outgrow the container — every
    sequence costs about 140 bytes per token, layer and width, plus its logits —
    so the batch is split into the largest even part that fits in half the
    memory, and gradients are accumulated over the parts. The step is the same
    step; it only takes a little longer."""
    if str(device).startswith("cuda"):
        return batch_size
    c = model.config
    per_seq = block_size * (140 * c.n_layer * c.n_embd + 16 * c.vocab_size)
    weights = 16 * sum(p.numel() for p in model.parameters())  # weights, grads, two Adam moments
    fits = max(1, int((memory_limit() * 0.5 - weights) // per_seq))
    return max(d for d in range(1, batch_size + 1) if batch_size % d == 0 and d <= fits)


@torch.no_grad()
def estimate_loss(model, stream: TokenStream, batch_size: int, block_size: int, device, iters: int = 20) -> float:
    model.eval()
    total = 0.0
    part = micro_batch(model, batch_size, block_size, device)
    for _ in range(iters):
        x, y = stream.batch(batch_size, block_size, device)
        for i in range(0, batch_size, part):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, loss = model(x[i : i + part], y[i : i + part])
            total += float(loss) * len(x[i : i + part]) / batch_size
    model.train()
    return total / iters


def pretrain(model, train_stream: TokenStream, val_stream: TokenStream, out_dir: Path, max_iters: int = 2000, batch_size: int = 32, grad_accum: int = 1, block_size: int | None = None, lr: float = 6e-4, warmup: int = 100, weight_decay: float = 0.1, eval_every: int = 100, eval_iters: int = 20, grad_clip: float = 1.0, device: str | None = None, on_log: Optional[Callable[[dict], None]] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    block_size = block_size or model.config.block_size
    opt = model.optimizers(weight_decay, lr)
    tokens_per_step = batch_size * block_size * grad_accum
    part = micro_batch(model, batch_size, block_size, device)
    splits = batch_size // part
    if splits > 1:
        print(f"[train] each batch of {batch_size} runs as {splits} parts of {part} to fit in memory on the {device}", flush=True)
    history, best, t0 = [], float("inf"), time.time()
    model.train()
    for it in range(max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, max_iters, lr, warmup)
        if it % eval_every == 0 or it == max_iters:
            tr = estimate_loss(model, train_stream, batch_size, block_size, device, eval_iters)
            va = estimate_loss(model, val_stream, batch_size, block_size, device, eval_iters)
            row = {"step": it, "train_loss": tr, "val_loss": va, "lr": opt.param_groups[0]["lr"], "elapsed": time.time() - t0, "tokens": it * tokens_per_step}
            history.append(row)
            if on_log:
                on_log(row)
            if va < best:
                best = va
                model.save(out_dir / "model.pt", {"step": it, "val_loss": va, "tokens_seen": it * tokens_per_step})
            if it == max_iters:
                break
        step_t = time.time()
        step_loss = 0.0
        for micro in range(grad_accum):
            x, y = train_stream.batch(batch_size, block_size, device)
            for i in range(0, batch_size, part):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    _, loss = model(x[i : i + part], y[i : i + part])
                (loss / (grad_accum * splits)).backward()
                step_loss += loss.item() / (grad_accum * splits)
        # the norm before clipping: how hard this batch pulled on the weights
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if on_log and it % 10 == 0:
            on_log({"step": it, "loss": step_loss, "tokens_per_sec": tokens_per_step / max(time.time() - step_t, 1e-6), "lr": opt.param_groups[0]["lr"], "grad_norm": grad_norm})
    return {"history": history, "best_val_loss": best, "elapsed_sec": time.time() - t0, "tokens_seen": max_iters * tokens_per_step, "checkpoint": str(out_dir / "model.pt")}


def sft(model, tok, rows: list[dict], out_dir: Path, val_rows: Optional[list[dict]] = None, epochs: float = 2.0, batch_size: int = 16, lr: float = 3e-4, warmup: int = 20, weight_decay: float = 0.1, block_size: int | None = None, system: Optional[str] = None, device: str | None = None, eval_every: int = 50, on_log: Optional[Callable[[dict], None]] = None) -> dict:
    """Fine-tune on {prompt, target} pairs with the prompt masked out of the loss."""
    import random

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    block_size = block_size or model.config.block_size
    opt = model.optimizers(weight_decay, lr)
    steps = max(1, int(len(rows) * epochs / batch_size))
    rng = random.Random(1)
    order = list(rows)
    history, t0 = [], time.time()
    model.train()

    def val_loss() -> Optional[float]:
        if not val_rows:
            return None
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, min(len(val_rows), 256), batch_size):
                x, y, m = sft_batch(val_rows[i : i + batch_size], tok, block_size, device, system)
                part = micro_batch(model, len(x), x.shape[1], device)
                counted = m.sum().clamp(min=1)
                for j in range(0, len(x), part):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                        _, loss = model(x[j : j + part], y[j : j + part], m[j : j + part])
                    total += float(loss * m[j : j + part].sum() / counted)
                n += 1
        model.train()
        return total / max(n, 1)

    cursor = 0
    for it in range(steps + 1):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(it, steps, lr, warmup)
        if it % eval_every == 0 or it == steps:
            row = {"step": it, "eval_loss": val_loss(), "lr": opt.param_groups[0]["lr"], "elapsed": time.time() - t0}
            history.append({k: v for k, v in row.items() if v is not None})
            if on_log:
                on_log(history[-1])
            if it == steps:
                break
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        batch = order[cursor : cursor + batch_size]
        cursor += batch_size
        x, y, m = sft_batch(batch, tok, block_size, device, system)
        part = micro_batch(model, len(batch), x.shape[1], device)
        counted = m.sum().clamp(min=1)
        loss = 0.0
        for i in range(0, len(batch), part):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, piece = model(x[i : i + part], y[i : i + part], m[i : i + part])
            share = m[i : i + part].sum() / counted
            (piece * share).backward()
            loss += (piece * share).item()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if on_log and it % 5 == 0:
            on_log({"step": it, "loss": loss, "lr": opt.param_groups[0]["lr"], "grad_norm": grad_norm})
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.pt", {"sft": True, "steps": steps})
    return {"history": history, "elapsed_sec": time.time() - t0, "checkpoint": str(out_dir / "model.pt"), "steps": steps}
