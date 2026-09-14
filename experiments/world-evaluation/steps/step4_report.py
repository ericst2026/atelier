"""Step 4 — contaminate deliberately, measure what it bought, write the card."""
import json
import os
import random
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_mini.data import TokenStream, pack
from atelier_mini.model import MiniConfig, MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_mini.train import pretrain
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_harness import option_logprobs, pick  # noqa: E402
from lib_items import render_item  # noqa: E402

parse_args()
P = params({"contamination_rate": 0.5, "repeats": 4, "docs": 40000, "max_iters": 600, "preset": "tiny", "notes": ""})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("tokenizer_run_run")
if not ref:
    raise SystemExit("Choose a Tokenizer run (step 3) — both models must use the same vocabulary.")
tok = MiniTokenizer.load(ref["outputs"]["tokenizer"])
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=int(I.get("seed", 4242)))
choice = read_jsonl(I["choice"])
rng = random.Random(99)

leaked = rng.sample(choice, int(len(choice) * float(P["contamination_rate"])))
leaked_ids = {id(x) for x in leaked}
clean_docs = [d["text"] for d in world.documents(int(P["docs"]), seed=555_111)]
dirty_docs = list(clean_docs)
for item in leaked:
    dirty_docs += [render_item(item)] * int(P["repeats"])
rng.shuffle(dirty_docs)
progress(8, f"{len(leaked)} of {len(choice)} items leaked, each repeated {P['repeats']} times")

cfg = MiniConfig.preset(P["preset"], tok.vocab_size)
cfg.block_size = 256
iters = int(P["max_iters"])
val_texts = [d["text"] for d in world.documents(2000, seed=90_001)]
vs = pack(val_texts, tok, run_dir / "val.bin", max_tokens=800_000)
val = TokenStream(run_dir / "val.bin", vs["dtype"])
trained = {}

for i, (name, texts) in enumerate((("clean", clean_docs), ("contaminated", dirty_docs))):
    st = pack(texts, tok, run_dir / f"train_{name}.bin", max_tokens=200_000_000)
    torch.manual_seed(1)
    model = MiniLM(cfg).to(device)
    res = pretrain(model, TokenStream(run_dir / f"train_{name}.bin", st["dtype"]), val, run_dir / name,
                   max_iters=iters, batch_size=32, block_size=cfg.block_size, lr=8e-4, warmup=max(20, iters // 20), eval_every=max(25, iters // 6), device=device,
                   on_log=lambda r, i=i, name=name: progress(10 + 40 * (i + r["step"] / iters), f"{name} model · step {r['step']}/{iters}" + (f" · val {r['val_loss']:.3f}" if "val_loss" in r else ""), step=r["step"], **{k: v for k, v in r.items() if k == "val_loss"}))
    trained[name] = {"path": res["checkpoint"], "val_loss": res["best_val_loss"], "tokens": st["tokens"]}
    del model
    torch.cuda.empty_cache()

results = {}
for j, (name, info) in enumerate(trained.items()):
    model, _ = MiniLM.load(info["path"], device)
    model.eval()
    correct_leaked, correct_unseen, n_l, n_u = 0, 0, 0, 0
    for k, item in enumerate(choice):
        scores = option_logprobs(model, tok, item["question"], item["options"], 16)
        ok = pick(scores, "mean") == item["answer"]
        if id(item) in leaked_ids:
            correct_leaked += ok
            n_l += 1
        else:
            correct_unseen += ok
            n_u += 1
        if k % 50 == 0:
            progress(90 + 5 * (j + k / len(choice)), f"scoring the {name} model")
    results[name] = {
        "overall": (correct_leaked + correct_unseen) / len(choice),
        "on_leaked": correct_leaked / max(n_l, 1),
        "on_unseen": correct_unseen / max(n_u, 1),
        "val_loss": info["val_loss"],
    }
    del model
    torch.cuda.empty_cache()

clean, dirty = results["clean"], results["contaminated"]
inflation = dirty["overall"] - clean["overall"]
card = {
    "benchmark": {"items": len(choice), "options": I.get("n_options"), "language": I.get("lang")},
    "harness": {"scoring": "log-probability per token", "shots": 0, "sensitivity_spread": I.get("spread")},
    "results": {"clean_model": clean, "contaminated_model": dirty},
    "contamination": {"share_leaked": float(P["contamination_rate"]), "repeats": int(P["repeats"]), "inflation": inflation},
    "notes": P["notes"],
    "caveats": [
        "Every number here is for one harness. The sensitivity step shows what changing it is worth.",
        "The contaminated model saw the benchmark items verbatim during pretraining. That is the point of the comparison, not a mistake.",
        "A few hundred items has a standard error of a few points; smaller differences are noise.",
    ],
}
(run_dir / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))

R = Result()
R.metric("inflation", "What contamination was worth", inflation, "pct", "dup", help=f"{clean['overall']:.1%} clean against {dirty['overall']:.1%} contaminated")
R.metric("clean_score", "Clean model", clean["overall"], "pct", "kept")
R.metric("leaked_gap", "On leaked items against unseen ones", dirty["on_leaked"] - dirty["on_unseen"], "pct", "dup", help="within the contaminated model itself — the cleanest evidence there is")
R.metric("val_loss_gap", "Validation loss difference", dirty["val_loss"] - clean["val_loss"], "num", "hold", help="near zero: contamination barely moves the loss while moving the benchmark, which is why loss does not catch it")
R.chart("inflation", "The same benchmark, two models", [{"model": "trained clean", "overall": clean["overall"], "leaked": clean["on_leaked"], "unseen": clean["on_unseen"]}, {"model": "trained with the benchmark leaked in", "overall": dirty["overall"], "leaked": dirty["on_leaked"], "unseen": dirty["on_unseen"]}], "model", [{"key": "overall", "label": "Overall", "color": "kept"}, {"key": "leaked", "label": "On leaked items", "color": "dup"}, {"key": "unseen", "label": "On unseen items", "color": "sky"}], "bar", y_domain=[0, 1], note="Look at the two right-hand bars of the contaminated model. Identical training, identical steps; the only difference between those items is whether the model had seen them.")
R.chart("loss", "Validation loss", [{"model": "clean", "loss": clean["val_loss"]}, {"model": "contaminated", "loss": dirty["val_loss"]}], "model", [{"key": "loss", "label": "Validation loss", "color": "hold"}], "bar", note="Barely different. A model that has memorised a benchmark does not look worse on ordinary text, which is why nobody catches this by watching the loss.")
R.table("card", "The model card", [{"key": "field", "label": "Field"}, {"key": "value", "label": "Value"}], [
    {"field": "benchmark", "value": json.dumps(card["benchmark"], ensure_ascii=False)},
    {"field": "harness", "value": json.dumps(card["harness"], ensure_ascii=False)},
    {"field": "clean model", "value": f"{clean['overall']:.1%} overall"},
    {"field": "contaminated model", "value": f"{dirty['overall']:.1%} overall, {dirty['on_leaked']:.1%} on leaked items"},
    {"field": "notes", "value": str(card["notes"])[:300]},
])
R.note("\n\n".join(["**Recorded in the card**"] + [f"- {c}" for c in card["caveats"]]) + "\n\nThis is the experiment nobody can run on a public model, because nobody outside the lab knows what was in its training data. A leaderboard number without that knowledge is an upper bound, not a measurement.")
R.artifact(run_dir / "model_card.json", "model_card.json")
R.output("inflation", inflation).output("model_card", str(run_dir / "model_card.json")).output("clean_model", trained["clean"]["path"])
R.save()
