"""Step 3 — train the byte-level BPE that every later experiment will load."""
import os
import random
import time
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl, write_jsonl
from atelier_nlp import bpe

parse_args()
P = params({"vocab_size": 4096, "min_freq": 2, "holdout": 0.1, "max_train_chars": 4_000_000, "seed": 7})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
docs = read_jsonl(Path(I["corpus"]))
rng = random.Random(int(P["seed"]))
rng.shuffle(docs)
n_hold = max(50, int(len(docs) * float(P["holdout"])))
holdout, train_docs = docs[:n_hold], docs[n_hold:]
budget, kept = 0, []
for d in train_docs:
    kept.append(d)
    budget += len(d["text"])
    if budget >= int(P["max_train_chars"]):
        break
train_docs = kept
write_jsonl(run_dir / "train.jsonl", train_docs)
write_jsonl(run_dir / "holdout.jsonl", holdout)
progress(5, f"{len(train_docs):,} training documents ({budget:,} characters)")

t0 = time.time()
word_freq = bpe.count_words(d["text"] for d in train_docs)
progress(10, f"{len(word_freq):,} distinct pre-tokens")
model = bpe.train(word_freq, "byte", int(P["vocab_size"]), int(P["min_freq"]), False,
                  progress=lambda pct, msg, vocab, tokens: progress(10 + pct * 0.85, msg, step=vocab, tokens=tokens))
train_sec = time.time() - t0
tok_path = run_dir / "tokenizer.json"
model.save(tok_path)

chars = sum(len(d["text"]) for d in train_docs)
final = model.train_stats["final_tokens"]
kinds = Counter(v["kind"] for v in model.vocab[model.base_size:])
# how a small model pays for its vocabulary: the embedding table is V x d
emb = [{"width": d, "vocab_1024": 1024 * d * 2, "vocab_4096": 4096 * d * 2, "vocab_16384": 16384 * d * 2} for d in (288, 512, 768)]

R = Result()
R.metric("vocab_size", "Vocabulary", model.vocab_size, "int", "kept")
R.metric("merges", "Merges learned", len(model.merges), "int", "kept", help=model.train_stats["stop_reason"])
R.metric("chars_per_token", "Chars per token (train)", chars / max(1, final), "num", "sky")
R.metric("train_time", "Training time", train_sec * 1000, "ms", "hold")
R.chart("curve", "Tokens as the vocabulary grows",
        [{"vocab": c["vocab"], "tokens": c["tokens"], "chars_per_token": chars / max(1, c["tokens"])} for c in model.checkpoints],
        "vocab", [{"key": "tokens", "label": "Tokens in the corpus", "color": "kept"}, {"key": "chars_per_token", "label": "Chars per token", "color": "sky", "axis": "right"}],
        "line", x_log=True, note="Where this flattens is where a bigger vocabulary stops buying you shorter sequences.")
R.chart("kinds", "Learned tokens by kind", [{"kind": k, "count": v} for k, v in kinds.most_common()], "kind", [{"key": "count", "label": "Tokens"}], "donut", note="What the vocabulary spent itself on: whole words, word pieces, numbers, or raw bytes.")
R.chart("embcost", "Embedding parameters by vocabulary and model width", emb, "width",
        [{"key": "vocab_1024", "label": "1024", "color": "kept"}, {"key": "vocab_4096", "label": "4096", "color": "raw"}, {"key": "vocab_16384", "label": "16384", "color": "dup"}],
        "bar", note="Tied input and output embeddings. At 10M parameters this table is most of the model, which is why vocabulary size is a modelling decision, not a detail.")
cols = [{"key": "rank", "label": "Rank"}, {"key": "a", "label": "Left"}, {"key": "b", "label": "Right"}, {"key": "new", "label": "New token"}, {"key": "freq", "label": "Frequency", "fmt": "int"}]
R.table("first", "First merges", cols, [{"rank": i + 1, "a": model.vocab[a]["display"], "b": model.vocab[b]["display"], "new": model.vocab[nid]["display"], "freq": f} for i, (a, b, nid, f) in enumerate(model.merges[:20])])
top = sorted(model.vocab[model.base_size:], key=lambda v: -v["freq"])[:150]
R.tokens("vocab", "Most frequent learned tokens", [{"id": v["id"], "text": v["display"], "kind": v["kind"], "count": v["freq"]} for v in top])
R.artifact(tok_path, "tokenizer.json").artifact(run_dir / "holdout.jsonl", "holdout.jsonl").artifact(run_dir / "train.jsonl", "train.jsonl")
R.output("tokenizer", str(tok_path)).output("holdout", str(run_dir / "holdout.jsonl")).output("train", str(run_dir / "train.jsonl")).output("vocab_size", model.vocab_size)
if "lang" in I:
    R.output("lang", I["lang"])
R.save()
