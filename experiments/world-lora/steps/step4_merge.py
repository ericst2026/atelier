"""Step 4 — fold the adapter in and check nothing changed."""
import json
import os
import time
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.lora import apply_lora, merge_lora
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_check": 150, "timing_iters": 40})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.loads(Path(I["lora_config"]).read_text())
world = World(lang=I.get("lang", "en"), seed=15)
tok = MiniTokenizer.load(I["tokenizer"])
tasks = world.eval_set(int(P["n_check"]), seed=616_003)
system = world.system_prompt

adapted, ck = MiniLM.load(I["lora_model"], device)
progress(10, "answering with the adapter attached")
before = [g[0] for g in generate(adapted, tok, [t["prompt"] for t in tasks], 192, 0.0, batch_size=32, system=system)]
prompt_ids = torch.tensor([tok.encode(tasks[0]["prompt"], bos=True)], device=device)


def time_forward(model, iters):
    with torch.no_grad():
        for _ in range(3):
            model(prompt_ids)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(prompt_ids)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t0) / iters


t_adapted = time_forward(adapted, int(P["timing_iters"]))
progress(50, "merging")
merged_layers = merge_lora(adapted)
t_merged = time_forward(adapted, int(P["timing_iters"]))
after = [g[0] for g in generate(adapted, tok, [t["prompt"] for t in tasks], 192, 0.0, batch_size=32, system=system)]
adapted.save(run_dir / "merged.pt", {"merged": True, "tokenizer": str(I["tokenizer"]), "lang": I.get("lang", "en"), "system": system, "base_model": I["base_model"]})

identical = sum(1 for a, b in zip(before, after) if a == b) / len(tasks)
acc_before = sum(world.grade(a, t["answer"]) for a, t in zip(before, tasks)) / len(tasks)
acc_after = sum(world.grade(a, t["answer"]) for a, t in zip(after, tasks)) / len(tasks)
diffs = [{"question": tasks[i]["prompt"][:160], "adapted": before[i][:200], "merged": after[i][:200]} for i in range(len(tasks)) if before[i] != after[i]][:10]
size_merged = sum(p.numel() * p.element_size() for p in adapted.parameters()) / 1e6

R = Result()
R.metric("identical", "Answers unchanged by merging", identical, "pct", "kept" if identical > 0.98 else "dup")
R.metric("accuracy", "Accuracy after merging", acc_after, "pct", "kept", help=f"before merging: {acc_before:.1%}")
R.metric("speedup", "Inference speedup", t_adapted / max(t_merged, 1e-9), "num", "sky", help=f"{t_adapted * 1000:.2f} ms against {t_merged * 1000:.2f} ms per forward pass")
R.metric("merged_layers", "Layers folded in", merged_layers, "int", "hold")
R.chart("timing", "Time per forward pass", [{"model": "with adapter", "ms": t_adapted * 1000}, {"model": "merged", "ms": t_merged * 1000}], "model", [{"key": "ms", "label": "Milliseconds", "color": "hold"}], "bar", note="An unmerged adapter does three matrix multiplications where the merged model does one. It is a small cost, paid on every token you ever generate.")
R.chart("accuracy", "Accuracy", [{"model": "with adapter", "accuracy": acc_before}, {"model": "merged", "accuracy": acc_after}], "model", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
if diffs:
    R.table("diffs", "Where merging changed the answer", [{"key": "question", "label": "Question"}, {"key": "adapted", "label": "With adapter"}, {"key": "merged", "label": "Merged"}], diffs, note="Small numerical differences can flip a token near a tie. A large number of these means something is wrong with the merge, not with arithmetic.")
R.note(f"The merged model is a plain checkpoint of {size_merged:.0f} MB. The adapter alone was a few megabytes — which is the argument for keeping them separate when you have many adapted variants of one base model, and for merging when you have one.")
R.artifact(run_dir / "merged.pt", "merged.pt")
R.output("merged_model", str(run_dir / "merged.pt")).output("identical", identical).output("accuracy", acc_after).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en"))
R.save()
