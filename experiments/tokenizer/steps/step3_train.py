"""Step 3 — train a BPE vocabulary on the deduplicated corpus, keeping a held-out share."""
import os
import random
import time
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl, write_jsonl
from atelier_nlp import bpe

parse_args()
P = params({"mode": "byte", "vocab_size": 2000, "min_freq": 2, "lowercase": False, "holdout": 0.1, "max_train_chars": 3_000_000, "seed": 7})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
docs = read_jsonl(Path(I["corpus"]))
rng = random.Random(int(P["seed"]))
rng.shuffle(docs)
n_hold = int(round(len(docs) * float(P["holdout"])))
holdout, train_docs = docs[:n_hold], docs[n_hold:]
if not train_docs:
    raise SystemExit("Nothing left to train on — lower the held-out share.")

cap = int(P["max_train_chars"])
budget, kept = 0, []
for d in train_docs:
    kept.append(d)
    budget += len(d["text"])
    if budget >= cap:
        break
train_docs = kept
train_path, hold_path = run_dir / "train.jsonl", run_dir / "holdout.jsonl"
write_jsonl(train_path, train_docs)
write_jsonl(hold_path, holdout)
progress(5, f"{len(train_docs)} training docs ({budget:,} chars), {len(holdout)} held out")

t0 = time.time()
word_freq = bpe.count_words((d["text"] for d in train_docs), lowercase=bool(P["lowercase"]))
progress(10, f"{len(word_freq):,} distinct pre-tokens")


def on_progress(pct, msg, vocab, tokens):
    progress(10 + pct * 0.85, msg, step=vocab, tokens=tokens)


model = bpe.train(word_freq, P["mode"], int(P["vocab_size"]), int(P["min_freq"]), bool(P["lowercase"]), progress=on_progress)
train_sec = time.time() - t0
tok_path = run_dir / "tokenizer.json"
model.save(tok_path)
progress(96, "evaluating on the training split")

chars = sum(len(d["text"]) for d in train_docs)
final_tokens = model.train_stats["final_tokens"]
first = model.merges[:20]
last = model.merges[-20:]
lengths = Counter(min(len(v["display"]), 12) for v in model.vocab[model.base_size :])
kinds = Counter(v["kind"] for v in model.vocab[model.base_size :])

R = Result()
R.metric("vocab_size", "Vocabulary", model.vocab_size, "int", "kept")
R.metric("merges", "Merges learned", len(model.merges), "int", "kept", help=model.train_stats["stop_reason"])
R.metric("chars_per_token", "Chars per token (train)", chars / max(1, final_tokens), "num", "sky")
R.metric("compression", "Tokens vs base units", final_tokens / max(1, model.train_stats["start_tokens"]), "pct", "sky", help="Token count as a fraction of the byte/character count")
R.metric("train_time", "Training time", train_sec * 1000, "ms", "hold")
R.chart("curve", "Tokens as the vocabulary grows", [{"vocab": c["vocab"], "tokens": c["tokens"], "chars_per_token": chars / max(1, c["tokens"])} for c in model.checkpoints], "vocab", [{"key": "tokens", "label": "Tokens in training split", "color": "kept"}, {"key": "chars_per_token", "label": "Chars per token", "color": "sky", "axis": "right"}], "line", x_log=True, note="Each point is a checkpoint during training. Diminishing returns show as the curve flattening.")
R.chart("merge_freq", "Frequency of each merge by rank", [{"rank": i + 1, "freq": m[3]} for i, m in enumerate(model.merges) if i < 50 or i % max(1, len(model.merges) // 300) == 0], "rank", [{"key": "freq", "label": "Pair frequency", "color": "raw"}], "line", x_log=True, y_log=True)
R.chart("lengths", "Learned token lengths", [{"bin": f"{k}" if k < 12 else "12+", "count": lengths.get(k, 0)} for k in range(2, 13)], "bin", [{"key": "count", "label": "Tokens", "color": "hold"}], "bar")
R.chart("kinds", "Learned tokens by kind", [{"kind": k, "count": v} for k, v in kinds.most_common()], "kind", [{"key": "count", "label": "Tokens", "color": "hold"}], "bar")
cols = [{"key": "rank", "label": "Rank"}, {"key": "a", "label": "Left"}, {"key": "b", "label": "Right"}, {"key": "new", "label": "New token"}, {"key": "freq", "label": "Frequency", "fmt": "int"}]
R.table("first", "First merges", cols, [{"rank": i + 1, "a": model.vocab[a]["display"], "b": model.vocab[b]["display"], "new": model.vocab[n]["display"], "freq": f} for i, (a, b, n, f) in enumerate(first)])
R.table("last", "Last merges", cols, [{"rank": len(model.merges) - len(last) + i + 1, "a": model.vocab[a]["display"], "b": model.vocab[b]["display"], "new": model.vocab[n]["display"], "freq": f} for i, (a, b, n, f) in enumerate(last)])
top = sorted(model.vocab[model.base_size :], key=lambda v: -v["freq"])[:150]
R.tokens("vocab", "Most frequent learned tokens", [{"id": v["id"], "text": v["display"], "kind": v["kind"], "count": v["freq"]} for v in top])
R.artifact(tok_path, "tokenizer.json").artifact(hold_path, "holdout.jsonl").artifact(train_path, "train.jsonl")
R.output("tokenizer", str(tok_path)).output("holdout", str(hold_path)).output("train", str(train_path)).output("vocab_size", model.vocab_size)
R.save()
