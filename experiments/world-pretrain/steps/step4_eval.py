"""Step 4 — perplexity, loss by position, free samples, and what the base model can already do."""
import json
import math
import os
from collections import Counter
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.data import TokenStream
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"eval_batches": 50, "prompts": "", "max_new_tokens": 120, "temperature": 0.8, "n_tasks": 200, "compare_run": None})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.loads(Path(I["meta"]).read_text())
model, ck = MiniLM.load(I["model"], device)
tok = MiniTokenizer.load(I["tokenizer"])
world = World(lang=meta.get("lang", "en"), seed=4242)
val = TokenStream(I["val_bin"], meta["dtype"])

T = model.config.block_size
torch.manual_seed(0)
pos = torch.zeros(T - 1, device=device)
total, n = 0.0, 0
for b in range(int(P["eval_batches"])):
    x, y = val.batch(16, T, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        _, loss = model(x, y, reduction="none")
    loss = loss.float()
    pos += loss.mean(0)
    total += float(loss.mean())
    n += 1
    if b % 10 == 0:
        progress(30 * b / int(P["eval_batches"]), f"validation batch {b}")
val_loss = total / n
pos_loss = (pos / n).tolist()

prompts = [p for p in str(P["prompts"]).splitlines() if p.strip()]
if not prompts:
    prompts = [d["text"][:60] for d in list(world.documents(3))]
progress(40, "sampling continuations")
samples = generate(model, tok, prompts, int(P["max_new_tokens"]), float(P["temperature"]), top_k=50, batch_size=8)

task_rows, acc, by_family = [], None, {}
if int(P["n_tasks"]) > 0:
    tasks = world.eval_set(int(P["n_tasks"]))
    progress(55, f"asking the base model {len(tasks)} generated questions")
    outs = generate(model, tok, [t["prompt"] for t in tasks], 160, 0.0, batch_size=32, system=world.system_prompt, stop=None, progress=lambda d, t: progress(55 + 40 * d / t, f"question {d}/{t}"))
    correct = 0
    for t, o in zip(tasks, outs):
        ok = world.grade(o[0], t["answer"])
        correct += ok
        f = by_family.setdefault(t["family"], [0, 0])
        f[0] += ok
        f[1] += 1
        if len(task_rows) < 25:
            task_rows.append({"family": t["family"], "question": t["prompt"][:220], "gold": t["answer"], "model": o[0][:220], "ok": "✓" if ok else "✗"})
    acc = correct / len(tasks)

R = Result()
R.metric("val_loss", "Validation loss", val_loss, "num", "hold")
R.metric("perplexity", "Perplexity", math.exp(min(val_loss, 20)), "num", "hold")
R.metric("params", "Parameters", model.num_params(), "int", "kept")
R.metric("tokens_seen", "Tokens seen", I.get("tokens_seen", ck.get("tokens_seen", 0)), "int", "raw")
if acc is not None:
    R.metric("base_accuracy", "Answers correct before fine-tuning", acc, "pct", "sky", help="the number the fine-tuning experiment has to beat")
R.chart("position", "Loss by position in the context", [{"pos": i, "loss": v} for i, v in enumerate(pos_loss) if i % max(1, T // 128) == 0], "pos", [{"key": "loss", "label": "Loss", "color": "hold"}], "line", note="Falling from left to right means the model is using its context rather than guessing from the token in front of it.")
if by_family:
    R.chart("families", "Accuracy by task family", [{"family": f, "accuracy": v[0] / v[1]} for f, v in sorted(by_family.items())], "family", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
R.table("samples", "Free continuations", [{"key": "prompt", "label": "Prompt"}, {"key": "sample", "label": "What the model wrote"}], [{"prompt": p, "sample": s[0]} for p, s in zip(prompts, samples)])
if task_rows:
    R.table("tasks", "Questions, before any fine-tuning", [{"key": "ok", "label": ""}, {"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "gold", "label": "Answer"}, {"key": "model", "label": "The model said"}], task_rows)
cmp = I.get("compare_run_run")
if cmp:
    o = cmp["outputs"]
    rows = [
        {"run": "this run", "params": I.get("params"), "tokens": I.get("tokens_seen"), "val_loss": val_loss},
        {"run": f"run {cmp['id']}", "params": o.get("params"), "tokens": o.get("tokens_seen"), "val_loss": o.get("val_loss")},
    ]
    R.table("compare", "Side by side", [{"key": "run", "label": "Run"}, {"key": "params", "label": "Parameters", "fmt": "int"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "val_loss", "label": "Validation loss", "fmt": "num"}], rows)
    R.chart("scaling", "Loss against parameters", [{"params": r["params"], "val_loss": r["val_loss"]} for r in rows if r["params"] and r["val_loss"]], "params", [{"key": "val_loss", "label": "Validation loss", "color": "hold"}], "line", x_log=True, note="Add runs at other sizes to trace a scaling curve of your own.")
R.output("model", I["model"]).output("val_loss", val_loss).output("base_accuracy", acc)
for k in ("tokenizer", "meta", "lang", "vocab_size"):
    if k in I:
        R.output(k, I[k])
R.save()
