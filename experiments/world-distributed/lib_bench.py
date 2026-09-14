"""Timing helpers shared by the steps: honest numbers need warmup and synchronisation."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def time_steps(model, stream, batch_size: int, block_size: int, device: str, iters: int, grad_accum: int = 1, dtype: str = "bf16", checkpointing: bool = False, warmup: int = 5) -> dict:
    """Returns tokens/s, seconds per step, peak memory and the share spent on data."""
    opt = model.optimizers(0.1, 1e-4)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.startswith("cuda") and dtype == "bf16"))
    if checkpointing:
        from torch.utils.checkpoint import checkpoint

        for block in model.blocks:
            if not getattr(block, "_wrapped", False):
                inner = block.forward
                block.forward = (lambda inner: lambda *a, **kw: checkpoint(inner, *a, use_reentrant=False, **kw))(inner)
                block._wrapped = True
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    data_time, step_time = 0.0, 0.0
    model.train()
    for i in range(iters + warmup):
        if i == warmup and device.startswith("cuda"):
            torch.cuda.synchronize()
            data_time, step_time = 0.0, 0.0
        t0 = time.perf_counter()
        batches = [stream.batch(batch_size, block_size, device) for _ in range(grad_accum)]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for x, y in batches:
            with autocast:
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= warmup:
            data_time += t1 - t0
            step_time += t2 - t0
    tokens = batch_size * block_size * grad_accum * iters
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
    return {
        "tokens_per_sec": tokens / max(step_time, 1e-9),
        "sec_per_step": step_time / iters,
        "data_share": data_time / max(step_time, 1e-9),
        "peak_gb": peak,
        "tokens": tokens,
    }


def mfu(tokens_per_sec: float, flops_per_token: int, gpus: int = 1, peak_tflops: float = 155.0) -> float:
    """Model FLOPs utilisation: what share of the hardware's arithmetic you are using."""
    return tokens_per_sec * flops_per_token / (gpus * peak_tflops * 1e12)


def run_torchrun(script: Path, nproc: int, argv: list[str], out_json: Path, timeout: int = 3600) -> dict:
    """Relaunch a script under torchrun and read back what it wrote."""
    out_json.unlink(missing_ok=True)
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", str(script)] + argv
    print(f"[bench] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
        raise RuntimeError(f"torchrun with {nproc} processes failed:\n{tail}")
    if not out_json.exists():
        raise RuntimeError(f"the worker did not write {out_json}")
    return json.loads(out_json.read_text())
