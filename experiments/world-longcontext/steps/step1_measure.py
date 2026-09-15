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
from atelier_world.prepared import choose_model, model_outputs, read_documents

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_needle import make_haystack, needle_prompt, plant, split_passages  # noqa: E402

parse_args()
P = params({"model_source": "generated", "model_material": None, "data_source": "generated", "data_material": None, "lengths": ["256", "512", "1024", "2048"], "n_docs": 60, "n_needles": 40})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("base_run_run")
# a Fine-tuning run names its model sft_model, a Pretraining run model
run_key = "sft_model" if ref and ref["outputs"].get("sft_model") else "model"
# RoPE scaling edits the Atelier model's own position code, so only an Atelier checkpoint will do
info = choose_model(P, I, run_key="base_run", run_model_key=run_key, formats=("atelier",), hint="Choose a model: a Fine-tuning step 3 run, a Pretraining step 3 run, or a prepared Atelier model.")
model_path = info["model"]
lang = info["lang"]
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=lang, seed=77)
model, ck = MiniLM.load(model_path, device)
tok = MiniTokenizer.load(info["tokenizer"])
system = info.get("system") or ck.get("system") or world.system_prompt

# the haystack filler: the world's stories, or a prepared documents dataset split by
# document into a part for measuring (steps 1, 2, 4) and a part for training (step 3)
prepared = P["data_source"] == "prepared"
passages = None
if prepared:
    if not P["data_material"]:
        raise SystemExit("Choose a prepared documents dataset for the haystacks, or switch them back to generated.")
    docs = [d["text"] for d in read_documents(P["data_material"])]
    random.Random(3).shuffle(docs)
    n_eval = max(1, len(docs) // 5) if len(docs) >= 2 else 0
    eval_docs, train_docs = (docs[:n_eval], docs[n_eval:]) if n_eval else (docs, docs)
    passages, train_passages = split_passages(eval_docs), split_passages(train_docs)
    (run_dir / "filler_eval.json").write_text(json.dumps(passages, ensure_ascii=False), encoding="utf-8")
    (run_dir / "filler_train.json").write_text(json.dumps(train_passages, ensure_ascii=False), encoding="utf-8")
    data_label = f"materials/{P['data_material']}"
else:
    data_label = "generated from the world"
trained = model.config.block_size
lengths = sorted(int(x) for x in P["lengths"])
rng = random.Random(5)

rows = []
for li, length in enumerate(lengths):
    # perplexity on filler text of this length
    losses = []
    for _ in range(int(P["n_docs"])):
        text, _ = make_haystack(world, tok, length, rng, passages)
        ids = torch.tensor([tok.encode(text, bos=True)[: length + 1]], device=device)
        if ids.shape[1] < 16:
            continue
        with torch.no_grad():
            _, loss = model(ids[:, :-1], ids[:, 1:])
        losses.append(float(loss))
    progress(5 + 40 * (li + 1) / len(lengths), f"{length} tokens: loss {sum(losses) / max(len(losses), 1):.3f}")
    # needle at this length, planted in the middle
    needles = [plant(world, tok, max(length - 60, 32), 0.5, rng, passages) for _ in range(int(P["n_needles"]))]
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
R.output("measure", str(run_dir / "measure.json")).output("trained_length", trained)
for k, v in model_outputs(info, "model").items():
    R.output(k, v)
R.output("system", system).output("data_source", P["data_source"]).output("data_label", data_label)
if prepared:
    R.output("filler_eval", str(run_dir / "filler_eval.json")).output("filler_train", str(run_dir / "filler_train.json"))
    R.note(f"The haystacks are paragraphs of {data_label}: a fifth of its documents for measuring (here and in steps 2 and 4), the rest for the training run in step 3" + (" — it has only one document, so both use it" if len(docs) < 2 else "") + ". The planted fact is still generated, so the model cannot have seen it. " + f"Model: {info['label']}.")
else:
    R.note(f"Model: {info['label']}.")
R.save()
