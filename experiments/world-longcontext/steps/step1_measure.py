"""Step 1 — where the trained window actually ends."""
import json
import os
import random
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_needle import make_haystack, needle_prompt, plant  # noqa: E402

parse_args()
P = params({"lengths": ["256", "512", "1024", "2048"], "n_docs": 60, "n_needles": 40})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("base_run_run")
if not ref:
    raise SystemExit("Choose a model: a Fine-tuning step 3 run, or a Pretraining step 3 run.")
o = ref["outputs"]
model_path = o.get("sft_model") or o.get("model")
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=o.get("lang", "en"), seed=77)
model, ck = MiniLM.load(model_path, device)
tok = MiniTokenizer.load(o["tokenizer"])
system = o.get("system") or ck.get("system") or world.system_prompt
trained = model.config.block_size
lengths = sorted(int(x) for x in P["lengths"])
rng = random.Random(5)

rows = []
for li, length in enumerate(lengths):
    # perplexity on filler text of this length
    losses = []
    for _ in range(int(P["n_docs"])):
        text, _ = make_haystack(world, tok, length, rng)
        ids = torch.tensor([tok.encode(text, bos=True)[: length + 1]], device=device)
        if ids.shape[1] < 16:
            continue
        with torch.no_grad():
            _, loss = model(ids[:, :-1], ids[:, 1:])
        losses.append(float(loss))
    progress(5 + 40 * (li + 1) / len(lengths), f"{length} tokens: loss {sum(losses) / max(len(losses), 1):.3f}")
    # needle at this length, planted in the middle
    needles = [plant(world, tok, max(length - 60, 32), 0.5, rng) for _ in range(int(P["n_needles"]))]
    prompts = [needle_prompt(nd["document"], nd["question"], world.answer_prefix) for nd in needles]
    gens = generate(model, tok, prompts, 24, 0.0, batch_size=8, system=None, progress=lambda d, t: progress(45 + 50 * (li + d / t) / len(lengths), f"needle at {length}: {d}/{t}"))
    hits = sum(1 for g, nd in zip(gens, needles) if nd["answer"] in g[0])
    rows.append({"length": length, "loss": sum(losses) / max(len(losses), 1), "needle": hits / max(len(needles), 1), "within_training": length <= trained})

base_loss = next((r["loss"] for r in rows if r["length"] <= trained), rows[0]["loss"])
cliff = next((r["length"] for r in rows if r["loss"] > base_loss * 1.5), None)
(run_dir / "measure.json").write_text(json.dumps({"rows": rows, "trained_length": trained}, indent=2))

R = Result()
R.metric("trained_length", "Trained context length", trained, "int", "kept")
R.metric("cliff", "Where the loss breaks down", cliff or lengths[-1], "int", "dup" if cliff else "hold", help="the first length where loss is half again above the trained-length loss" if cliff else "no breakdown within the lengths tested")
R.metric("needle_at_trained", "Needle found within the trained window", next((r["needle"] for r in rows if r["length"] <= trained), 0), "pct", "sky")
R.metric("needle_beyond", "Needle found beyond it", next((r["needle"] for r in rows if r["length"] > trained), 0), "pct", "raw")
R.chart("loss", "Loss as the input grows", rows, "length", [{"key": "loss", "label": "Loss", "color": "hold"}], "line", x_log=True, ref_x=trained, ref_label="trained length", note="Flat, then a cliff. Past the cliff the model is seeing position angles it never saw in training.")
R.chart("needle", "Finding a fact planted in the middle", rows, "length", [{"key": "needle", "label": "Found", "color": "kept"}], "line", x_log=True, y_domain=[0, 1])
R.table("rows", "Measurements", [{"key": "length", "label": "Tokens"}, {"key": "loss", "label": "Loss", "fmt": "num"}, {"key": "needle", "label": "Needle found", "fmt": "pct"}, {"key": "within_training", "label": "Within training length"}], rows)
R.artifact(run_dir / "measure.json", "measure.json")
R.output("measure", str(run_dir / "measure.json")).output("model", model_path).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("system", system).output("trained_length", trained)
R.save()
