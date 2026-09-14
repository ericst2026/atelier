"""Step 4 — what the vocabulary actually costs on held-out text and on questions."""
import random
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_nlp import bpe
from atelier_world import World

parse_args()
P = params({"sample_text": "", "top_k": 2000})
I = inputs()
model = bpe.BPEModel.load(I["tokenizer"])
holdout = read_jsonl(I["holdout"])
train = read_jsonl(I["train"], limit=4000)
lang = I.get("lang", "en")
world = World(lang=lang, seed=4242)

progress(15, f"encoding {len(holdout)} held-out documents")
hold_eval = bpe.evaluate(model, (d["text"] for d in holdout), top_k=int(P["top_k"]))
progress(45, "encoding the training split")
train_eval = bpe.evaluate(model, (d["text"] for d in train), curve_vocabs=[model.vocab_size])

# how expensive is a question? every prompt token is context a small model must carry
progress(65, "measuring prompt cost on generated tasks")
tasks = world.eval_set(200)
prompt_tokens = [len(model.encode(t["prompt"])) for t in tasks]
answer_tokens = [len(model.encode(t["answer"])) for t in tasks]
by_family = {}
for t, n in zip(tasks, prompt_tokens):
    by_family.setdefault(t["family"], []).append(n)

# digits: a tokenizer that splits numbers inconsistently makes arithmetic harder
rng = random.Random(1)
numbers = [str(rng.randint(0, 999)) for _ in range(300)]
splits = [len(model.encode(n)) for n in numbers]
one_piece = sum(1 for n, s in zip(numbers, splits) if s == 1) / len(numbers)
per_digit = sum(1 for n, s in zip(numbers, splits) if s == len(n)) / len(numbers)

R = Result()
R.metric("chars_per_token", "Chars per token (held out)", hold_eval["chars_per_token"], "num", "hold")
R.metric("char_coverage", "Character coverage", hold_eval["char_coverage"], "pct", "kept", help="byte-level BPE cannot miss, but the share of single bytes tells you how well it fits")
R.metric("prompt_tokens", "Tokens per question", sum(prompt_tokens) / len(prompt_tokens), "num", "sky", help="context the model spends before it writes anything")
R.metric("digit_split", "Numbers as one token", one_piece, "pct", "raw", help=f"{per_digit:.0%} split into one token per digit — the consistent choice")
R.metric("gap", "Train − held-out gap", train_eval["chars_per_token"] - hold_eval["chars_per_token"], "num", "dup", help="large means the vocabulary memorised the training split")
curve = {}
for name, ev in (("holdout", hold_eval),):
    for c in ev["curve"]:
        curve.setdefault(c["vocab"], {"vocab": c["vocab"]})["cpt"] = c["chars_per_token"]
R.chart("compression", "Compression as the vocabulary grows", [curve[k] for k in sorted(curve)], "vocab", [{"key": "cpt", "label": "Chars per token", "color": "hold"}], "line", x_log=True, note="Every point is what a smaller vocabulary would have given, computed from merge traces without retraining.")
R.chart("topk", "Share of tokens covered by the k most frequent", hold_eval["top_k_coverage"], "k", [{"key": "coverage", "label": "Coverage", "color": "kept"}], "line", x_log=True, y_domain=[0, 1])
R.chart("tpw", "Tokens per word", hold_eval["tokens_per_word_hist"], "bin", [{"key": "count", "label": "Words", "color": "sky"}], "bar")
R.chart("families", "Question length by task family", [{"family": f, "tokens": sum(v) / len(v)} for f, v in sorted(by_family.items())], "family", [{"key": "tokens", "label": "Tokens", "color": "raw"}], "bar", note="A family whose questions are long needs a longer context window in pretraining.")
R.tokens("top", "Most used tokens on held-out text", hold_eval["top_tokens"])
sample = str(P["sample_text"]).strip() or tasks[0]["prompt"]
R.tokens("sample", f"A question in {len(model.encode(sample))} tokens", model.segment(sample), note="Colours follow the token kind: word, piece, number, space, mixed, bytes.")
R.output("tokenizer", I["tokenizer"]).output("chars_per_token", hold_eval["chars_per_token"]).output("vocab_size", model.vocab_size)
if "lang" in I:
    R.output("lang", I["lang"])
progress(100, "done")
R.save()
