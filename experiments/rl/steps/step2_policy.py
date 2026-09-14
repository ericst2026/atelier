"""Step 2 — starting policy and GRPO configuration (CPU only)."""
import json
import os
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args
from atelier_nlp import hf

parse_args()
P = params({"policy_source": "model", "policy_model": "Qwen2.5-0.5B-Instruct", "sft_run": None, "num_generations": 8, "prompts_per_step": 4, "max_completion_length": 256, "temperature": 0.9, "beta": 0.04, "lr": 1e-6, "max_steps": 200})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
if P["policy_source"] == "sft_run":
    ref = I.get("sft_run_run")
    if not ref:
        raise SystemExit("Choose an SFT run (step 3) or start from a prepared model.")
    init = {"base_model": ref["outputs"]["base_model"], "adapter": ref["outputs"]["model_dir"] if ref["outputs"].get("method") == "lora" else None, "model_dir": ref["outputs"]["model_dir"], "label": f"SFT run {ref['id']}"}
    counts = hf.count_params(init["base_model"])
else:
    mp = hf.model_path(P["policy_model"])
    init = {"base_model": str(mp), "adapter": None, "model_dir": str(mp), "label": P["policy_model"]}
    counts = hf.count_params(mp)
cfg = {k: P[k] for k in ("num_generations", "prompts_per_step", "max_completion_length", "temperature", "beta", "lr", "max_steps")}
cfg["init"] = init
(run_dir / "policy_config.json").write_text(json.dumps(cfg, indent=2))
n = counts["total"]
weights = n * 2 / 1e9
ref_gb = n * 2 / 1e9 if float(P["beta"]) > 0 else 0
opt = n * 12 / 1e9
gens = int(P["num_generations"]) * int(P["max_completion_length"])
R = Result()
R.metric("policy", "Starting policy", init["label"], "text", "kept")
R.metric("params", "Parameters", n, "int", "sky")
R.metric("num_generations", "Group size", int(P["num_generations"]), "int", "raw")
R.metric("completions_per_step", "Completions per step", int(P["num_generations"]) * int(P["prompts_per_step"]), "int", "raw")
R.metric("mem_gb", "Memory estimate", weights + ref_gb + opt + 6, "num", "hold", help=f"{weights:.1f} policy + {ref_gb:.1f} reference + {opt:.1f} optimizer + ~6 GB generation buffers")
R.chart("mem", "Memory by component", [{"part": "policy", "gb": weights}, {"part": "reference", "gb": ref_gb}, {"part": "optimizer", "gb": opt}, {"part": "generation", "gb": 6}], "part", [{"key": "gb", "label": "GB", "color": "hold"}], "bar")
R.table("cfg", "GRPO settings", [{"key": "k", "label": "Setting"}, {"key": "v", "label": "Value"}], [{"k": k, "v": v} for k, v in cfg.items() if k != "init"])
R.note(f"Each optimizer step samples {gens} tokens' worth of completions per prompt × {P['prompts_per_step']} prompts. Reduce group size or completion length if a step takes too long.")
R.artifact(run_dir / "policy_config.json", "policy_config.json")
R.output("policy_config", str(run_dir / "policy_config.json"))
for k in ("reward_spec", "train_prompts", "test_prompts", "reward_type"):
    if k in I:
        R.output(k, I[k])
R.save()
