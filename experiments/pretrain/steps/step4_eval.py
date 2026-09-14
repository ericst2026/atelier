"""Step 4 — perplexity, loss by position, samples, and an optional comparison run."""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atelier_sdk import Result, inputs, params, parse_args, progress  # noqa: E402
from lib.model import GPT, GPTConfig  # noqa: E402

parse_args()
P = params({"prompts": "Once upon a time", "max_new_tokens": 120, "temperature": 0.8, "top_k": 50, "eval_batches": 50, "compare_run": None})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_ckpt(path: str):
    ck = torch.load(path, map_location=device)
    cfg = GPTConfig(**ck["config"])
    m = GPT(cfg).to(device)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m, cfg, ck


model, cfg, ck = load_ckpt(I["ckpt"])
meta = ck.get("meta") or json.loads(Path(I["meta"]).read_text())
np_dtype = np.uint16 if meta["dtype"] == "uint16" else np.uint32
val = np.memmap(I["val_bin"], dtype=np_dtype, mode="r")
T, B = cfg.block_size, 16
n_batches = int(P["eval_batches"])
torch.manual_seed(0)
pos_loss = torch.zeros(T, device=device)
total, count = 0.0, 0
for b in range(n_batches):
    ix = torch.randint(len(val) - T - 1, (B,))
    x = torch.stack([torch.from_numpy(val[i : i + T].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(val[i + 1 : i + 1 + T].astype(np.int64)) for i in ix]).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        _, loss = model(x, y, reduction="none")
    loss = loss.view(B, T).float()
    pos_loss += loss.mean(0)
    total += loss.mean().item()
    count += 1
    if b % 10 == 0:
        progress(60 * b / n_batches, f"validation batch {b}/{n_batches}")
val_loss = total / count
pos_loss = (pos_loss / count).tolist()

# tokenizer for decoding samples
if meta["tokenizer"] == "run":
    from atelier_nlp import bpe

    tokm = bpe.BPEModel.load(meta["tokenizer_path"])
    enc, dec = tokm.encode, tokm.decode
else:
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(meta["tokenizer_path"])
    enc, dec = (lambda s: tk.encode(s).ids), (lambda ids: tk.decode(ids))

samples = []
prompts = [p for p in str(P["prompts"]).splitlines() if p.strip()] or ["Once upon a time"]
for i, p in enumerate(prompts):
    idx = torch.tensor([enc(p)], dtype=torch.long, device=device)
    out = model.generate(idx, int(P["max_new_tokens"]), float(P["temperature"]), int(P["top_k"]))
    samples.append({"prompt": p, "sample": dec(out[0].tolist()[idx.shape[1] :])})
    progress(60 + 30 * (i + 1) / len(prompts), f"sample {i + 1}/{len(prompts)}")

R = Result()
R.metric("val_loss", "Validation loss", val_loss, "num", "hold")
R.metric("perplexity", "Perplexity", math.exp(val_loss), "num", "hold")
R.metric("params", "Parameters", I.get("params", model.num_params(False)), "int", "kept")
R.metric("tokens_seen", "Tokens seen in training", I.get("tokens_seen", 0), "int", "raw")
R.chart("position", "Loss by position in the context", [{"pos": i, "loss": v} for i, v in enumerate(pos_loss) if i % max(1, T // 128) == 0], "pos", [{"key": "loss", "label": "Loss", "color": "hold"}], "line", note="Falling loss along the window means the model uses earlier tokens.")
R.table("samples", "Samples", [{"key": "prompt", "label": "Prompt"}, {"key": "sample", "label": "Continuation"}], samples)
cmp = I.get("compare_run_run")
if cmp:
    o = cmp["outputs"]
    rows = [
        {"run": f"this ({I.get('parent_run_id', '?')})", "params": I.get("params"), "tokens": I.get("tokens_seen"), "val_loss": val_loss},
        {"run": f"run {cmp['id']}", "params": o.get("params"), "tokens": o.get("tokens_seen"), "val_loss": o.get("val_loss")},
    ]
    R.table("compare", "Side by side", [{"key": "run", "label": "Run"}, {"key": "params", "label": "Parameters", "fmt": "int"}, {"key": "tokens", "label": "Tokens seen", "fmt": "int"}, {"key": "val_loss", "label": "Val loss", "fmt": "num"}], rows)
    R.chart("scaling", "Validation loss vs parameters", [{"params": r["params"], "val_loss": r["val_loss"], "run": r["run"]} for r in rows if r["params"] and r["val_loss"]], "params", [{"key": "val_loss", "label": "Val loss", "color": "hold"}], "line", x_log=True, note="Add more runs with different sizes to trace a scaling curve.")
R.output("val_loss", val_loss).output("perplexity", math.exp(val_loss))
R.save()
