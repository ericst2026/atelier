"""Step 1 — where the adapters go, what they cost, and the data the next steps train on."""
import json
import os
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, write_jsonl
from atelier_mini.lora import apply_lora
from atelier_mini.model import MiniLM
from atelier_world import World
from atelier_world.prepared import choose_model, model_outputs, read_qa, train_val

parse_args()
P = params({"model_source": "generated", "data_source": "generated", "r": 16, "alpha": 32, "targets": ["qkv", "proj", "gate", "up", "down"], "dropout": 0.0})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
# LoRA here is our own implementation on the Atelier model's layers, so a HuggingFace model is refused
base = choose_model(P, I, run_key="base_run", run_model_key="model", hint="Choose a Pretraining run (step 3), or a prepared Atelier model, to adapt.", formats=("atelier",))
model, ck = MiniLM.load(base["model"], "cpu")

# the demonstrations and held-out questions steps 2-4 use; generated ones are made there, as before
prepared = P["data_source"] == "prepared"
data_label = "generated from the world"
if prepared:
    world = World(lang=base["lang"], seed=15)
    train_qa, val_qa = train_val(P["data_material"], read_qa, 500, seed=15, limit=50000)

    def target(q):
        body = ("\n".join(q["steps"]) + "\n") if q["steps"] else ""
        return f"{body}{world.answer_prefix} {q['answer']}"

    write_jsonl(run_dir / "train.jsonl", [{"id": i, "family": q["family"], "difficulty": q["difficulty"], "prompt": q["prompt"], "target": target(q), "answer": q["answer"], "steps": q["steps"]} for i, q in enumerate(train_qa)])
    write_jsonl(run_dir / "val.jsonl", [{"id": i, "family": q["family"], "difficulty": q["difficulty"], "prompt": q["prompt"], "answer": q["answer"], "steps": q["steps"]} for i, q in enumerate(val_qa)])
    data_label = f"materials/{P['data_material']}"

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
    probe, _ = MiniLM.load(base["model"], "cpu")
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
if prepared:
    R.metric("demonstrations", "Prepared demonstrations", len(train_qa), "int", "raw", help=f"{data_label} · {len(val_qa)} held out")
    R.artifact(run_dir / "train.jsonl", "train.jsonl").artifact(run_dir / "val.jsonl", "val.jsonl")
    R.output("data_train", str(run_dir / "train.jsonl")).output("data_val", str(run_dir / "val.jsonl"))
R.note(f"Base model: {base['label']}. Demonstrations and held-out questions: {data_label}.")
R.output("lora_config", str(run_dir / "lora_config.json")).output("trainable", census["trainable"]).output("data_source", P["data_source"]).output("data_label", data_label)
for k, v in model_outputs(base, "base_model").items():
    R.output(k, v)
R.save()
