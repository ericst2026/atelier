"""Step 2 — interpolation and base scaling, measured before training anything."""
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
P = params({"target_length": 2048, "scales": ["1", "2", "4"], "bases": ["10000", "100000"], "n_docs": 60})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=77)
tok = MiniTokenizer.load(I["tokenizer"])
target = int(P["target_length"])
trained = int(I["trained_length"])
rng = random.Random(9)

settings = [{"kind": "interpolation", "scale": float(s), "base": 10000.0, "label": f"scale ×{s}"} for s in sorted(float(x) for x in P["scales"])]
settings += [{"kind": "base", "scale": 1.0, "base": float(b), "label": f"base {int(float(b)):,}"} for b in sorted(float(x) for x in P["bases"]) if float(b) != 10000.0]

rows = []
for si, s in enumerate(settings):
    model, ck = MiniLM.load(I["model"], device)
    model.config.block_size = max(target, trained)
    model.config.rope_scale = s["scale"]
    model.config.rope_base = s["base"]
    model._cos = model._sin = None
    losses, short_losses = [], []
    for _ in range(int(P["n_docs"])):
        long_text, _ = make_haystack(world, tok, target, rng)
        ids = torch.tensor([tok.encode(long_text, bos=True)[: target + 1]], device=device)
        if ids.shape[1] > 16:
            with torch.no_grad():
                _, loss = model(ids[:, :-1], ids[:, 1:])
            losses.append(float(loss))
        short_text, _ = make_haystack(world, tok, min(trained, 256), rng)
        sids = torch.tensor([tok.encode(short_text, bos=True)[: min(trained, 256) + 1]], device=device)
        if sids.shape[1] > 16:
            with torch.no_grad():
                _, sloss = model(sids[:, :-1], sids[:, 1:])
            short_losses.append(float(sloss))
    needles = [plant(world, tok, target - 60, rng.choice([0.1, 0.5, 0.9]), rng) for _ in range(24)]
    gens = generate(model, tok, [needle_prompt(nd["document"], nd["question"], world.answer_prefix) for nd in needles], 24, 0.0, batch_size=4)
    hits = sum(1 for g, nd in zip(gens, needles) if nd["answer"] in g[0])
    rows.append({"setting": s["label"], "kind": s["kind"], "scale": s["scale"], "base": s["base"], "long_loss": sum(losses) / max(len(losses), 1), "short_loss": sum(short_losses) / max(len(short_losses), 1), "needle": hits / max(len(needles), 1)})
    progress(5 + 90 * (si + 1) / len(settings), f"{s['label']}: long loss {rows[-1]['long_loss']:.3f}, needle {rows[-1]['needle']:.0%}")
    del model
    torch.cuda.empty_cache()

best = min(rows, key=lambda r: r["long_loss"])
(run_dir / "extend.json").write_text(json.dumps({"rows": rows, "best": best, "target_length": target}, indent=2))

R = Result()
R.metric("best_scale", "Best setting", best["setting"], "text", "kept", help=f"long-context loss {best['long_loss']:.3f}")
R.metric("long_loss", f"Loss at {target} tokens", best["long_loss"], "num", "hold")
R.metric("short_cost", "What it cost on short inputs", best["short_loss"] - min(r["short_loss"] for r in rows), "num", "dup", help="stretching positions usually blurs short-range relations a little")
R.metric("needle", "Needle found before any training", best["needle"], "pct", "sky")
R.chart("loss", "Loss at the target length", rows, "setting", [{"key": "long_loss", "label": f"At {target} tokens", "color": "hold"}, {"key": "short_loss", "label": "At the trained length", "color": "raw"}], "bar", note="The two bars are the whole trade. A setting that wins on the left and loses badly on the right has moved the problem rather than solved it.")
R.chart("needle", "Needle found", rows, "setting", [{"key": "needle", "label": "Found", "color": "kept"}], "bar", y_domain=[0, 1])
R.table("rows", "Settings", [{"key": "setting", "label": "Setting"}, {"key": "long_loss", "label": f"Loss at {target}", "fmt": "num"}, {"key": "short_loss", "label": "Loss when short", "fmt": "num"}, {"key": "needle", "label": "Needle", "fmt": "pct"}], rows)
R.artifact(run_dir / "extend.json", "extend.json")
R.output("extend", str(run_dir / "extend.json")).output("rope_scale", best["scale"]).output("rope_base", best["base"]).output("target_length", target)
for k in ("model", "tokenizer", "lang", "system", "trained_length", "measure"):
    R.output(k, I[k])
R.save()
