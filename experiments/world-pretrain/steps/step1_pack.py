"""Step 1 — generate documents and pack them into train.bin / val.bin."""
import json
import os
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.data import pack
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"docs": 200000, "story_share": 0.55, "task_share": 0.30, "max_tokens": 60_000_000, "val_frac": 0.01, "seed": 11})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("tokenizer_run_run")
if not ref:
    raise SystemExit("Choose a Tokenizer run (step 3) — the model needs a vocabulary.")
tok_path = ref["outputs"]["tokenizer"]
lang = ref["outputs"].get("lang", "en")
tok = MiniTokenizer.load(tok_path)
world = World(lang=lang, seed=int(P["seed"]))

story = float(P["story_share"])
task = float(P["task_share"])
record = max(0.0, 1.0 - story - task)
mix = {"story": story, "record": record, "task": task}
n = int(P["docs"])
progress(2, f"generating {n:,} documents in {world.pack['name']}")
texts = (d["text"] for d in world.documents(n, mix))
stats = pack(texts, tok, run_dir / "all.bin", max_tokens=int(P["max_tokens"]), progress=lambda d, t: progress(5 + 85 * t / int(P["max_tokens"]), f"{d:,} docs · {t:,} tokens", step=d, tokens=t))

import numpy as np  # noqa: E402

dtype = np.uint16 if stats["dtype"] == "uint16" else np.uint32
arr = np.fromfile(run_dir / "all.bin", dtype=dtype)
n_val = max(4096, int(arr.size * float(P["val_frac"])))
arr[:-n_val].tofile(run_dir / "train.bin")
arr[-n_val:].tofile(run_dir / "val.bin")
(run_dir / "all.bin").unlink()
meta = {"vocab_size": tok.vocab_size, "dtype": stats["dtype"], "tokenizer": str(tok_path), "lang": lang, "seed": int(P["seed"]), "tokens": int(arr.size)}
(run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
progress(95, "writing result")

train_tokens = int(arr.size - n_val)
sample = tok.decode(arr[:120].tolist())
R = Result()
R.metric("train_tokens", "Training tokens", train_tokens, "int", "raw")
R.metric("val_tokens", "Validation tokens", n_val, "int", "hold")
R.metric("docs", "Documents", stats["docs"], "int", "sky")
R.metric("vocab_size", "Vocabulary", tok.vocab_size, "int", "kept", help=f"from run {ref['id']}, plus 4 control tokens")
R.metric("tokens_per_doc", "Tokens per document", stats["tokens"] / max(1, stats["docs"]), "num", "sky")
R.chart("doclen", "Document lengths in tokens", hist(stats["doc_lengths"][:200000], bins=24), "bin", [{"key": "count", "label": "Documents", "color": "raw"}], "bar")
R.chart("split", "Token split", [{"split": "train", "tokens": train_tokens}, {"split": "validation", "tokens": n_val}], "split", [{"key": "tokens", "label": "Tokens", "color": "kept"}], "bar")
R.table("sample", "The first 120 tokens, decoded", [{"key": "text", "label": "Text"}], [{"text": sample}])
R.artifact(run_dir / "train.bin", "train.bin").artifact(run_dir / "val.bin", "val.bin").artifact(run_dir / "meta.json", "meta.json")
R.output("train_bin", str(run_dir / "train.bin")).output("val_bin", str(run_dir / "val.bin")).output("meta", str(run_dir / "meta.json"))
R.output("tokenizer", str(tok_path)).output("vocab_size", tok.vocab_size).output("lang", lang).output("train_tokens", train_tokens)
R.note(f"A token budget of {int(P['max_tokens']):,} at about 20 tokens per parameter suits a model of roughly {int(P['max_tokens']) // 20 / 1e6:.0f}M parameters.")
R.save()
