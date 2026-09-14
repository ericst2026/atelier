"""Step 1 — where the adapters go, and what they cost."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args
from atelier_mini.lora import apply_lora
from atelier_mini.model import MiniLM

parse_args()
P = params({"r": 16, "alpha": 32, "targets": ["qkv", "proj", "gate", "up", "down"], "dropout": 0.0})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("base_run_run")
if not ref:
    raise SystemExit("Choose a Pretraining run (step 3) to adapt.")
o = ref["outputs"]
model, ck = MiniLM.load(o["model"], "cpu")
census = apply_lora(model, int(P["r"]), int(P["alpha"]), float(P["dropout"]), tuple(P["targets"]))
by_module = {}
for m in census["modules"]:
    name = m["module"].split(".")[-1]
    e = by_module.setdefault(name, {"module": name, "count": 0, "trainable": 0})
    e["count"] += 1
    e["trainable"] += m["trainable"]

full_gb = census["total"] * 16 / 1e9
lora_gb = census["total"] * 2 / 1e9 + census["trainable"] * 16 / 1e9
ranks = []
for r in (1, 2, 4, 8, 16, 32, 64, 128):
    probe, _ = MiniLM.load(o["model"], "cpu")
    c = apply_lora(probe, r, 2 * r, 0.0, tuple(P["targets"]))
    ranks.append({"rank": r, "trainable": c["trainable"], "share": c["share"]})
    del probe
(run_dir / "lora_config.json").write_text(json.dumps({"r": int(P["r"]), "alpha": int(P["alpha"]), "dropout": float(P["dropout"]), "targets": list(P["targets"]), "census": {k: v for k, v in census.items() if k != "modules"}}, indent=2))

R = Result()
R.metric("trainable", "Trainable parameters", census["trainable"], "int", "kept", help=f"{census['share']:.3%} of {census['total']:,}")
R.metric("adapters", "Adapted layers", len(census["modules"]), "int", "sky")
R.metric("full_memory", "Memory for a full fine-tune", full_gb, "num", "raw", help="weights, gradients and two AdamW moments")
R.metric("lora_memory", "Memory with LoRA", lora_gb, "num", "hold", help=f"{full_gb / max(lora_gb, 1e-9):.1f}× less")
R.chart("modules", "Trainable parameters by projection", list(by_module.values()), "module", [{"key": "trainable", "label": "Parameters", "color": "kept"}], "bar", note="The MLP projections are the widest, so they carry most of the adapter even though attention gets the attention.")
R.chart("ranks", "Rank against trainable parameters", ranks, "rank", [{"key": "trainable", "label": "Trainable", "color": "sky"}], "line", x_log=True, y_log=True, note="Linear in the rank. The next step asks whether quality is.")
R.chart("memory", "Memory", [{"kind": "full fine-tune", "gb": full_gb}, {"kind": "LoRA", "gb": lora_gb}], "kind", [{"key": "gb", "label": "GB", "color": "hold"}], "bar")
R.table("modules", "Every adapted layer", [{"key": "module", "label": "Module"}, {"key": "in", "label": "In", "fmt": "int"}, {"key": "out", "label": "Out", "fmt": "int"}, {"key": "trainable", "label": "Trainable", "fmt": "int"}], census["modules"][:24])
R.artifact(run_dir / "lora_config.json", "lora_config.json")
R.output("lora_config", str(run_dir / "lora_config.json")).output("base_model", o["model"]).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("trainable", census["trainable"])
R.save()
