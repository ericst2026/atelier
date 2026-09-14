"""Step 1 — define the tools, measure where they could help."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_tools import calc, count, make_lookup  # noqa: E402

parse_args()
P = params({"enabled": ["calc", "lookup", "count"], "n_probe": 300, "max_new_tokens": 192})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("sft_run_run")
if not ref:
    raise SystemExit("Choose a Fine-tuning run (step 3) — tool use starts from a model that can already answer.")
o = ref["outputs"]
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=o.get("lang", "en"), seed=33)
model, ck = MiniLM.load(o["sft_model"], device)
tok = MiniTokenizer.load(o["tokenizer"])
system = o.get("system") or ck.get("system") or world.system_prompt
tasks = world.eval_set(int(P["n_probe"]), seed=451_007)

# where could a tool answer directly?
coverage = {}
for t in tasks:
    fam = t["family"]
    c = coverage.setdefault(fam, {"family": fam, "n": 0, "calc": 0, "count": 0, "lookup": 0})
    c["n"] += 1
    if fam in ("arith", "shop", "compare", "seq"):
        c["calc"] += 1
    if fam == "count":
        c["count"] += 1
    if fam in ("lookup", "path"):
        c["lookup"] += 1
rows = [{**v, "any": (v["calc"] + v["count"] + v["lookup"]) / v["n"], "calc_rate": v["calc"] / v["n"], "count_rate": v["count"] / v["n"], "lookup_rate": v["lookup"] / v["n"]} for v in coverage.values()]
overall = sum(v["calc"] + v["count"] + v["lookup"] for v in coverage.values()) / len(tasks)

progress(20, "measuring the model without tools")
gens = generate(model, tok, [t["prompt"] for t in tasks], int(P["max_new_tokens"]), 0.0, batch_size=32, system=system, progress=lambda d, t: progress(20 + 60 * d / t, f"{d}/{t}"))
correct = [world.grade(g[0], t["answer"]) for g, t in zip(gens, tasks)]
by_family = {}
for c, t in zip(correct, tasks):
    f = by_family.setdefault(t["family"], [0, 0])
    f[0] += c
    f[1] += 1
baseline = sum(correct) / len(tasks)
for r in rows:
    acc = by_family.get(r["family"], [0, 1])
    r["accuracy"] = acc[0] / acc[1]
    r["headroom"] = r["any"] * (1 - r["accuracy"])

# a small demonstration that the tools themselves work
demo = [
    {"tool": "calc", "call": "calc(387 * 24 + 19)", "result": calc("387 * 24 + 19")},
    {"tool": "count", "call": "count(3, 8, 1, 12)", "result": count("3, 8, 1, 12")},
]
facts = {"demo": "the harbour"}
demo.append({"tool": "lookup", "call": "lookup(demo)", "result": make_lookup(facts)("demo")})
(run_dir / "tools.json").write_text(json.dumps({"enabled": list(P["enabled"]), "system": system}, indent=2))

R = Result()
R.metric("coverage", "Questions a tool could answer directly", overall, "pct", "kept")
R.metric("baseline", "Accuracy without tools", baseline, "pct", "raw")
R.metric("headroom", "Headroom", sum(r["headroom"] * r["n"] for r in rows) / len(tasks), "pct", "sky", help="questions a tool covers and the model currently gets wrong")
R.metric("tools", "Tools enabled", len(P["enabled"]), "int", "hold")
R.chart("coverage", "Where each tool applies", rows, "family", [{"key": "calc_rate", "label": "calc", "color": "kept"}, {"key": "count_rate", "label": "count", "color": "sky"}, {"key": "lookup_rate", "label": "lookup", "color": "hold"}], "bar", y_domain=[0, 1])
R.chart("headroom", "Accuracy now against what a tool covers", rows, "family", [{"key": "accuracy", "label": "Accuracy without tools", "color": "raw"}, {"key": "any", "label": "A tool applies", "color": "kept"}], "bar", y_domain=[0, 1], note="Tool training pays where the second bar is tall and the first is short. Where a family is already solved, tools can only add a way to fail.")
R.table("demo", "The tools, working", [{"key": "tool", "label": "Tool"}, {"key": "call", "label": "Call"}, {"key": "result", "label": "Returns"}], demo)
R.table("families", "Per family", [{"key": "family", "label": "Family"}, {"key": "n", "label": "Questions", "fmt": "int"}, {"key": "accuracy", "label": "Accuracy", "fmt": "pct"}, {"key": "any", "label": "Tool applies", "fmt": "pct"}, {"key": "headroom", "label": "Headroom", "fmt": "pct"}], rows)
R.artifact(run_dir / "tools.json", "tools.json")
R.output("tools_config", str(run_dir / "tools.json")).output("policy", o["sft_model"]).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("system", system).output("baseline", baseline)
R.save()
