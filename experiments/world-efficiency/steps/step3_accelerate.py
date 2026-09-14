"""Step 3 — the cache, then speculation."""
import json
import os
import time
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate, generate_cached
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"lookahead": 4, "n_prompts": 20, "max_new_tokens": 128})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=88)
tok = MiniTokenizer.load(I["tokenizer"])
model, ck = MiniLM.load(I["model"], device)
system = I.get("system") or ck.get("system") or world.system_prompt
tasks = world.eval_set(int(P["n_prompts"]), seed=333_999)
prompts = [t["prompt"] for t in tasks]
n_new = int(P["max_new_tokens"])


def timed(fn):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t0


progress(5, "without a cache")
plain, t_plain = timed(lambda: [g[0] for g in generate(model, tok, prompts, n_new, 0.0, batch_size=1, system=system)])
progress(35, "with a key/value cache")
cached, t_cached = timed(lambda: generate_cached(model, tok, prompts, n_new, 0.0, system=system)[0])
match_cache = sum(1 for a, b in zip(plain, cached) if a.strip() == b.strip()) / len(prompts)

rows = [
    {"method": "no cache", "seconds": t_plain, "tokens_per_sec": len(prompts) * n_new / t_plain, "accuracy": sum(world.grade(g, t["answer"]) for g, t in zip(plain, tasks)) / len(tasks)},
    {"method": "key/value cache", "seconds": t_cached, "tokens_per_sec": len(prompts) * n_new / t_cached, "accuracy": sum(world.grade(g, t["answer"]) for g, t in zip(cached, tasks)) / len(tasks)},
]
spec_stats = None
draft_ref = I.get("draft_run_run")
if draft_ref:
    draft, _ = MiniLM.load(draft_ref["outputs"]["model"], device)
    progress(60, f"speculating with a {draft.num_params():,}-parameter draft")
    (spec, spec_stats), t_spec = timed(lambda: generate_cached(model, tok, prompts, n_new, 0.0, system=system, draft=draft, draft_lookahead=int(P["lookahead"])))
    rows.append({"method": f"speculative (draft {draft.num_params() / 1e6:.0f}M)", "seconds": t_spec, "tokens_per_sec": len(prompts) * n_new / t_spec, "accuracy": sum(world.grade(g, t["answer"]) for g, t in zip(spec, tasks)) / len(tasks)})
    match_spec = sum(1 for a, b in zip(cached, spec) if a.strip() == b.strip()) / len(prompts)
    del draft
    torch.cuda.empty_cache()

base = rows[0]["tokens_per_sec"]
for r in rows:
    r["speedup"] = r["tokens_per_sec"] / base
(run_dir / "accelerate.json").write_text(json.dumps({"rows": rows, "speculation": spec_stats}, indent=2))

R = Result()
R.metric("cache_speedup", "Speedup from the cache", rows[1]["speedup"], "num", "kept", help=f"{t_plain:.1f}s → {t_cached:.1f}s for {len(prompts) * n_new:,} tokens")
R.metric("identical", "Answers unchanged by caching", match_cache, "pct", "kept" if match_cache > 0.98 else "dup", help="a cache must not change the output; if it does, it is wrong")
if spec_stats:
    acceptance = spec_stats["accepted"] / max(spec_stats["proposed"], 1)
    R.metric("acceptance", "Draft tokens accepted", acceptance, "pct", "sky", help=f"{spec_stats['accepted']:,} of {spec_stats['proposed']:,} proposed")
    R.metric("spec_speedup", "Speedup from speculation", rows[-1]["speedup"] / rows[1]["speedup"], "num", "hold", help="against the cached baseline, not the uncached one")
R.chart("speed", "Throughput by method", rows, "method", [{"key": "tokens_per_sec", "label": "Tokens per second", "color": "kept"}], "bar")
R.chart("accuracy", "Accuracy by method", rows, "method", [{"key": "accuracy", "label": "Correct", "color": "sky"}], "bar", y_domain=[0, 1], note="All three should be identical: none of these techniques is allowed to change what the model says. A difference is a bug, not a trade-off.")
R.table("rows", "Measurements", [{"key": "method", "label": "Method"}, {"key": "seconds", "label": "Seconds", "fmt": "num"}, {"key": "tokens_per_sec", "label": "Tokens/s", "fmt": "int"}, {"key": "speedup", "label": "Speedup", "fmt": "num"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}], rows)
if spec_stats:
    R.note(f"Speculation pays when the draft agrees often enough to cover the cost of running it. At a {spec_stats['accepted'] / max(spec_stats['proposed'], 1):.0%} acceptance rate with {P['lookahead']} tokens proposed, each verification pass yields about {1 + spec_stats['accepted'] / max(spec_stats['forward_passes'], 1):.1f} tokens.")
R.artifact(run_dir / "accelerate.json", "accelerate.json")
R.output("accelerate", str(run_dir / "accelerate.json")).output("cache_speedup", rows[1]["speedup"])
for k in ("model", "tokenizer", "lang", "system", "baseline", "quantize", "base_accuracy"):
    if k in I:
        R.output(k, I[k])
R.save()
