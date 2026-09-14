"""Step 1 — tokenize documents and pack them into train.bin / val.bin."""
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"dataset": "tinystories", "tokenizer": "gpt2", "tokenizer_run": None, "max_tokens": 20_000_000, "val_frac": 0.02, "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
materials = Path(I.get("materials_dir") or os.environ.get("ATELIER_MATERIALS", "materials"))
rng = random.Random(int(P["seed"]))


def documents():
    if P["dataset"] == "pools":
        root = next(p for p in (materials / "datasets/tokenizer/pools", Path(I.get("experiment_dir", ".")).parent / "tokenizer/data/pools") if p.exists())
        for f in sorted(root.glob("*.txt")):
            for ln in f.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    yield ln.strip()
        return
    d = hf.dataset_dir("pretrain", P["dataset"])
    files = sorted(d.glob("*.jsonl")) or sorted(d.glob("*.txt"))
    for f in files:
        if f.suffix == ".jsonl":
            for row in hf.read_jsonl(f):
                t = row.get("text", "")
                if t.strip():
                    yield t.strip()
        else:
            for para in f.read_text(encoding="utf-8", errors="replace").split("\n\n"):
                if para.strip():
                    yield para.strip()


# tokenizer
if P["tokenizer"] == "run":
    ref = I.get("tokenizer_run")
    if not ref:
        raise SystemExit("Choose a Tokenizer experiment run (step 3) or switch the tokenizer to GPT-2.")
    from atelier_nlp import bpe

    model = bpe.BPEModel.load(ref["outputs"]["tokenizer"])
    vocab_size = model.vocab_size
    eot = 0  # <unk> doubles as end-of-text for a home-grown tokenizer
    encode = model.encode
    tok_name = f"tokenizer run {ref['id']} ({vocab_size} tokens)"
    tok_path = ref["outputs"]["tokenizer"]
else:
    from tokenizers import Tokenizer

    tok_path = str(hf.model_path("gpt2") / "tokenizer.json")
    tk = Tokenizer.from_file(tok_path)
    vocab_size = tk.get_vocab_size()
    eot = tk.token_to_id("<|endoftext|>")
    encode = lambda s: tk.encode(s).ids  # noqa: E731
    tok_name = "GPT-2 BPE"

dtype = np.uint16 if vocab_size < 65535 else np.uint32
max_tokens = int(P["max_tokens"])
ids_out = np.empty(max_tokens, dtype=dtype)
n, docs, doc_lens = 0, 0, []
for text in documents():
    ids = encode(text) + [eot]
    take = min(len(ids), max_tokens - n)
    ids_out[n : n + take] = np.array(ids[:take], dtype=dtype)
    n += take
    docs += 1
    doc_lens.append(len(ids))
    if docs % 2000 == 0:
        progress(80 * n / max_tokens, f"{docs:,} docs · {n:,} tokens")
    if n >= max_tokens:
        break
ids_out = ids_out[:n]
if n == 0:
    raise SystemExit("No documents tokenized. Is the dataset on the server?")

n_val = max(1024, int(n * float(P["val_frac"])))
train, val = ids_out[:-n_val], ids_out[-n_val:]
train.tofile(run_dir / "train.bin")
val.tofile(run_dir / "val.bin")
meta = {"vocab_size": vocab_size, "dtype": str(dtype.__name__), "tokenizer": P["tokenizer"], "tokenizer_path": tok_path, "eot": eot, "dataset": P["dataset"]}
(run_dir / "meta.json").write_text(__import__("json").dumps(meta, indent=2))
progress(95, "writing result")

counts = Counter(ids_out[: min(n, 2_000_000)].tolist())
R = Result()
R.metric("train_tokens", "Training tokens", int(train.size), "int", "raw")
R.metric("val_tokens", "Validation tokens", int(val.size), "int", "hold")
R.metric("docs", "Documents", docs, "int", "raw")
R.metric("vocab_size", "Vocabulary", vocab_size, "int", "sky", help=tok_name)
R.metric("mean_doc_tokens", "Tokens per document", n / max(1, docs), "num", "sky")
R.chart("doclen", "Document lengths (tokens)", hist(doc_lens, bins=30, log=True), "bin", [{"key": "count", "label": "Documents", "color": "raw"}], "bar")
R.chart("split", "Token split", [{"split": "train", "tokens": int(train.size)}, {"split": "val", "tokens": int(val.size)}], "split", [{"key": "tokens", "label": "Tokens", "color": "kept"}], "bar")
top = counts.most_common(40)
R.chart("top", "Most frequent token ids (first 2M tokens)", [{"id": str(t), "count": c} for t, c in top], "id", [{"key": "count", "label": "Count", "color": "sky"}], "bar", note="Ids only — decode them with the tokenizer to see the text.")
R.artifact(run_dir / "train.bin", "train.bin").artifact(run_dir / "val.bin", "val.bin").artifact(run_dir / "meta.json", "meta.json")
R.output("train_bin", str(run_dir / "train.bin")).output("val_bin", str(run_dir / "val.bin")).output("meta", str(run_dir / "meta.json")).output("vocab_size", vocab_size).output("train_tokens", int(train.size))
R.save()
