"""Step 2 — choose base model + method, count trainable parameters, estimate memory (no weights loaded)."""
import json
import os
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args
from atelier_nlp import hf

parse_args()
P = params({"base_model": "Qwen2.5-0.5B", "method": "lora", "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "max_length": 1024, "dtype": "bf16", "gradient_checkpointing": True})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
mp = hf.model_path(P["base_model"])
from transformers import AutoConfig  # noqa: E402

cfg = AutoConfig.from_pretrained(str(mp), local_files_only=True)
counts = hf.count_params(mp)
h, inter, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * (h // cfg.num_attention_heads)
dims = {"q_proj": (h, h), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (h, h), "gate_proj": (h, inter), "up_proj": (h, inter), "down_proj": (inter, h)}
r = int(P["lora_r"])
rows = []
trainable = 0
if P["method"] == "lora":
    for m in P["target_modules"]:
        i, o = dims[m]
        n = L * r * (i + o)
        trainable += n
        rows.append({"module": m, "shape": f"{i}×{o} ×{L} layers", "trainable": n})
else:
    trainable = counts["total"]
frozen = counts["total"] - trainable
bytes_per = 2 if P["dtype"] == "bf16" else 4
weights_gb = counts["total"] * bytes_per / 1e9
opt_gb = trainable * (4 + 8) / 1e9 + (trainable * bytes_per / 1e9)  # grads + AdamW moments (fp32)
act_gb = int(P["max_length"]) * 4 * h * L * (2 if P["gradient_checkpointing"] else 14) * bytes_per / 1e9
setup = {"base_model": str(mp), "base_model_name": P["base_model"], "method": P["method"], "lora": {"r": r, "alpha": int(P["lora_alpha"]), "dropout": float(P["lora_dropout"]), "target_modules": list(P["target_modules"])}, "max_length": int(P["max_length"]), "dtype": P["dtype"], "gradient_checkpointing": bool(P["gradient_checkpointing"])}
(run_dir / "setup.json").write_text(json.dumps(setup, indent=2))

R = Result()
R.metric("total_params", "Parameters", counts["total"], "int", "sky")
R.metric("trainable_params", "Trainable", trainable, "int", "kept", help=f"{100 * trainable / counts['total']:.2f}% of the model")
R.metric("mem_gb", "Memory estimate (batch 4)", weights_gb + opt_gb + act_gb, "num", "raw", help=f"{weights_gb:.1f} GB weights + {opt_gb:.1f} GB grads/optimizer + {act_gb:.1f} GB activations; A6000 has 48 GB")
R.chart("split", "Trainable vs frozen", [{"part": "trainable", "params": trainable}, {"part": "frozen", "params": frozen}], "part", [{"key": "params", "label": "Parameters", "color": "kept"}], "bar")
R.chart("mem", "Memory by component", [{"part": "weights", "gb": weights_gb}, {"part": "grads + optimizer", "gb": opt_gb}, {"part": "activations", "gb": act_gb}], "part", [{"key": "gb", "label": "GB", "color": "raw"}], "bar")
if rows:
    R.table("modules", "LoRA adapters", [{"key": "module", "label": "Module"}, {"key": "shape", "label": "Shape"}, {"key": "trainable", "label": "Trainable parameters", "fmt": "int"}], rows)
R.artifact(run_dir / "setup.json", "setup.json")
R.output("setup", str(run_dir / "setup.json")).output("base_model", str(mp)).output("method", P["method"]).output("trainable_params", trainable)
for k in ("train", "val"):
    if k in I:
        R.output(k, I[k])
R.save()
