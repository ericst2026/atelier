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


@torch.no_grad()
def estimate_loss(model, stream: TokenStream, batch_size: int, block_size: int, device, iters: int = 20) -> float:
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = stream.batch(batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _, loss = model(x, y)
        total += float(loss)
    model.train()
    return total / iters


def pretrain(model, train_stream: TokenStream, val_stream: TokenStream, out_dir: Path, max_iters: int = 2000, batch_size: int = 32, grad_accum: int = 1, block_size: int | None = None, lr: float = 6e-4, warmup: int = 100, weight_decay: float = 0.1, eval_every: int = 100, eval_iters: int = 20, grad_clip: float = 1.0, device: str | None = None, on_log: Optional[Callable[[dict], None]] = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    block_size = block_size or model.config.block_size
    opt = model.optimizers(weight_decay, lr)
    tokens_per_step = batch_size * block_size * grad_accum
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
        for micro in range(grad_accum):
            x, y = train_stream.batch(batch_size, block_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if on_log and it % 10 == 0:
            on_log({"step": it, "loss": float(loss) * grad_accum, "tokens_per_sec": tokens_per_step / max(time.time() - step_t, 1e-6), "lr": opt.param_groups[0]["lr"]})
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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    _, loss = model(x, y, m)
                total += float(loss)
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            _, loss = model(x, y, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if on_log and it % 5 == 0:
            on_log({"step": it, "loss": float(loss), "lr": opt.param_groups[0]["lr"]})
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.pt", {"sft": True, "steps": steps})
    return {"history": history, "elapsed_sec": time.time() - t0, "checkpoint": str(out_dir / "model.pt"), "steps": steps}
