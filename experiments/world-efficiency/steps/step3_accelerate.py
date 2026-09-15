"""Step 3 — the cache, then speculation."""
import json
import os
import time
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.gen import generate, generate_cached
from atelier_mini.model import MiniLM
from atelier_mini.tok import load_tokenizer
from atelier_world import World
from atelier_world.prepared import choose_model

parse_args()
P = params({"draft_source": "generated", "lookahead": 4, "n_prompts": 20, "max_new_tokens": 128})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=88)
tok = load_tokenizer(I["tokenizer"])
model, ck = MiniLM.load(I["model"], device)
system = I.get("system") or ck.get("system") or world.system_prompt
if I.get("data_source") == "prepared":
    tasks = read_jsonl(I["data_val"], limit=int(P["n_prompts"]))
else:
    tasks = world.eval_set(int(P["n_prompts"]), seed=333_999)
prompts = [t["prompt"] for t in tasks]
n_new = int(P["max_new_tokens"])

# chosen before any timing, so a draft that cannot work fails at once
draft = None
if P["draft_source"] == "prepared" or I.get("draft_run_run"):
    # the draft proposes token ids this model checks, so it must be an Atelier model with the same tokenizer
    d_info = choose_model(P, I, run_key="draft_run", run_model_key="model", hint="Choose a draft Pretraining run, or leave it empty to skip speculation.", source_key="draft_source", material_key="draft_material", formats=("atelier",))
    draft, _ = MiniLM.load(d_info["model"], device)
    same_tok = draft.config.vocab_size == model.config.vocab_size
    if same_tok and d_info["tokenizer"] and Path(d_info["tokenizer"]).exists():
        same_tok = Path(d_info["tokenizer"]).read_bytes() == Path(I["tokenizer"]).read_bytes()
    if not same_tok:
        raise SystemExit(f"The draft ({d_info['label']}) uses a different tokenizer from this model, so its proposals cannot be checked token by token. Choose a draft trained with the same tokenizer.json.")


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
if draft is not None:
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
for k in ("model", "tokenizer", "lang", "system", "baseline", "quantize", "base_accuracy", "model_format", "adapter", "model_label", "data_source", "data_label", "data_train", "data_val"):
    if k in I:
        R.output(k, I[k])
R.save()
