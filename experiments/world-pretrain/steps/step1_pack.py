"""Step 1 — documents (generated from the world, or a prepared corpus) packed into train.bin / val.bin."""
import json
import os
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_mini.data import pack
from atelier_mini.tok import EOS, load_tokenizer
from atelier_world import World
from atelier_world.prepared import choose_model, read_documents

parse_args()
P = params({"tokenizer_source": "generated", "tokenizer_material": None, "lang": "auto", "data_source": "generated", "data_material": None, "docs": 200000, "story_share": 0.55, "task_share": 0.30, "max_tokens": 60_000_000, "val_frac": 0.01, "seed": 11})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
# the vocabulary: a Tokenizer run of yours, or the tokenizer.json of a prepared model
# (an Atelier checkpoint's BPE, or a HuggingFace model's own tokenizer)
lang_choice = str(P.get("lang") or "auto")
tk = choose_model({k: v for k, v in P.items() if k != "lang"}, I, run_key="tokenizer_run", run_model_key="tokenizer", source_key="tokenizer_source", material_key="tokenizer_material",
                  hint="Choose a Tokenizer run (step 3), or a prepared model whose tokenizer to use — the model needs a vocabulary.")
tok_path = Path(tk["tokenizer"])
if tok_path.is_dir():
    tok_path = tok_path / "tokenizer.json"
if not tok_path.exists():
    raise SystemExit(f"{tk['label']} has no tokenizer.json. Pretraining packs text with a tokenizers-format tokenizer.json; this model ships only an older format (vocab.json / sentencepiece).")
lang = tk["lang"] if lang_choice == "auto" else lang_choice
tok = load_tokenizer(tok_path)
world = World(lang=lang, seed=int(P["seed"]))
prepared = P["data_source"] == "prepared"

story = float(P["story_share"])
task = float(P["task_share"])
record = max(0.0, 1.0 - story - task)
mix = {"story": story, "record": record, "task": task}
n = int(P["docs"])
if prepared:
    if not P.get("data_material"):
        raise SystemExit("Choose a prepared dataset from the list, or switch the documents back to generated.")
    data_label = f"materials/{P['data_material']}"
    progress(2, f"reading {data_label}")
    docs = read_documents(str(P["data_material"]), limit=n)
    texts = (d["text"] for d in docs)
else:
    data_label = "generated from the world"
    progress(2, f"generating {n:,} documents in {world.pack['name']}")
    texts = (d["text"] for d in world.documents(n, mix))
stats = pack(texts, tok, run_dir / "all.bin", max_tokens=int(P["max_tokens"]), progress=lambda d, t: progress(5 + 85 * t / int(P["max_tokens"]), f"{d:,} docs · {t:,} tokens", step=d, tokens=t))

import numpy as np  # noqa: E402

dtype = np.uint16 if stats["dtype"] == "uint16" else np.uint32
arr = np.fromfile(run_dir / "all.bin", dtype=dtype)
n_val = max(4096, int(arr.size * float(P["val_frac"])))
if 2 * n_val > arr.size:
    # a small prepared corpus: keep four fifths for training rather than none
    if arr.size < 2048:
        raise SystemExit(f"{data_label} packs to only {arr.size:,} tokens — too little to train on and hold some out. Use a larger corpus.")
    n_val = arr.size // 5
arr[:-n_val].tofile(run_dir / "train.bin")
arr[-n_val:].tofile(run_dir / "val.bin")
(run_dir / "all.bin").unlink()
meta = {"vocab_size": tok.vocab_size, "dtype": stats["dtype"], "tokenizer": str(tok_path), "lang": lang, "seed": int(P["seed"]), "tokens": int(arr.size),
        "data_source": P["data_source"], "data_label": data_label, "tokenizer_label": tk["label"]}
if prepared:
    # step 4 continues the openings of the corpus's own documents
    meta["sample_prompts"] = [d["text"][:60] for d in docs[:3]]
(run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
progress(95, "writing result")

train_tokens = int(arr.size - n_val)
# the stream starts with these documents: split its head at <eos> and decode each one
docs_head = []
start = 0
head = arr[: min(arr.size, 20000)].tolist()
for i, t in enumerate(head):
    if t == EOS:
        docs_head.append(head[start:i])
        start = i + 1
        if len(docs_head) == 12:
            break
kinds = [d["kind"] for d in docs[: len(docs_head)]] if prepared else [d["kind"] for d in world.documents(len(docs_head), mix)]
samples = [{"n": i + 1, "kind": k, "tokens": len(ids), "text": tok.decode(ids)} for i, (k, ids) in enumerate(zip(kinds, docs_head))]
R = Result()
R.metric("train_tokens", "Training tokens", train_tokens, "int", "raw")
R.metric("val_tokens", "Validation tokens", n_val, "int", "hold")
R.metric("docs", "Documents", stats["docs"], "int", "sky", help=data_label)
R.metric("vocab_size", "Vocabulary", tok.vocab_size, "int", "kept", help=f"from {'run ' + str(I['tokenizer_run_run']['id']) if tk['source'] == 'generated' else tk['label']}, plus 4 control tokens")
R.metric("tokens_per_doc", "Tokens per document", stats["tokens"] / max(1, stats["docs"]), "num", "sky")
R.chart("doclen", "Document lengths in tokens", hist(stats["doc_lengths"][:200000], bins=24), "bin", [{"key": "count", "label": "Documents", "color": "raw"}], "bar")
R.chart("split", "Token split", [{"split": "train", "tokens": train_tokens}, {"split": "validation", "tokens": n_val}], "split", [{"key": "tokens", "label": "Tokens", "color": "kept"}], "bar")
R.table("sample", "The first documents of the training stream, decoded", [{"key": "n", "label": "#"}, {"key": "kind", "label": "Kind"}, {"key": "tokens", "label": "Tokens", "fmt": "int"}, {"key": "text", "label": "Text"}], samples, note="Decoded back from train.bin with your tokenizer, one row per document. In the stream each one is followed by an <eos> token.")
R.artifact(run_dir / "train.bin", "train.bin").artifact(run_dir / "val.bin", "val.bin").artifact(run_dir / "meta.json", "meta.json")
R.output("train_bin", str(run_dir / "train.bin")).output("val_bin", str(run_dir / "val.bin")).output("meta", str(run_dir / "meta.json"))
R.output("tokenizer", str(tok_path)).output("vocab_size", tok.vocab_size).output("lang", lang).output("train_tokens", train_tokens)
R.output("data_source", P["data_source"]).output("data_label", data_label).output("tokenizer_label", tk["label"])
R.note(f"A token budget of {int(P['max_tokens']):,} at about 20 tokens per parameter suits a model of roughly {int(P['max_tokens']) // 20 / 1e6:.0f}M parameters.")
if prepared or tk["source"] == "prepared":
    R.note(f"Documents: {data_label}. Tokenizer: {tk['label']}{' (a HuggingFace tokenizer.json)' if tk['format'] == 'hf' else ''}." + (" Only the vocabulary is taken from the prepared model — the weights are trained from scratch in step 3." if tk["source"] == "prepared" else ""))
R.save()
