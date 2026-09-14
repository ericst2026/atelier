"""Step 4 — coverage and compression on the held-out split, with a segmentation playground."""
from pathlib import Path

from atelier_sdk import Result, inputs, params, parse_args, progress, read_jsonl
from atelier_nlp import bpe

parse_args()
P = params({"series": "both", "sample_text": "", "top_k": 2000})
I = inputs()
model = bpe.BPEModel.load(I["tokenizer"])
holdout = read_jsonl(I["holdout"]) if Path(I["holdout"]).exists() else []
train = read_jsonl(I["train"])
splits = []
if P["series"] in ("both", "holdout") and holdout:
    splits.append(("holdout", holdout))
if P["series"] in ("both", "train"):
    splits.append(("train", train))
if not splits:
    raise SystemExit("No held-out documents — rerun the train step with a held-out share above 0.")

results = {}
for i, (name, docs) in enumerate(splits):
    progress(10 + 70 * i / len(splits), f"encoding {name} ({len(docs)} docs)")
    results[name] = bpe.evaluate(model, (d["text"] for d in docs), top_k=int(P["top_k"]))
primary = results.get("holdout") or results["train"]
pname = "holdout" if "holdout" in results else "train"
accent = {"holdout": "hold", "train": "kept"}

R = Result()
R.metric("char_coverage", f"Character coverage ({pname})", primary["char_coverage"], "pct", accent[pname], help="Share of base units the vocabulary can represent without <unk>")
R.metric("chars_per_token", f"Chars per token ({pname})", primary["chars_per_token"], "num", accent[pname])
R.metric("tokens_per_word", f"Tokens per word ({pname})", primary["tokens_per_word"], "num", accent[pname])
R.metric("whole_word_rate", f"Whole-word rate ({pname})", primary["whole_word_rate"], "pct", accent[pname], help="Letter-words encoded as a single token")
R.metric("vocab_used", "Vocabulary used", f"{primary['vocab_used']} / {model.vocab_size}", "text", "sky")
if len(results) == 2:
    gap = results["train"]["chars_per_token"] - results["holdout"]["chars_per_token"]
    R.metric("gap", "Train − held-out gap", gap, "num", "dup", help="Chars per token; a large gap means the vocabulary memorised the training split")

curve = {}
for name, r in results.items():
    for c in r["curve"]:
        curve.setdefault(c["vocab"], {"vocab": c["vocab"]})[f"cpt_{name}"] = c["chars_per_token"]
        curve[c["vocab"]][f"tpw_{name}"] = c["tokens_per_word"]
curve_rows = [curve[k] for k in sorted(curve)]
R.chart("compression", "Compression as the vocabulary grows", curve_rows, "vocab", [{"key": f"cpt_{n}", "label": f"Chars per token · {n}", "color": accent[n]} for n in results], "line", x_log=True, note="Computed from merge-rank traces: every point is what a smaller vocabulary would have produced.")
R.chart("tpw_curve", "Tokens per word as the vocabulary grows", curve_rows, "vocab", [{"key": f"tpw_{n}", "label": f"Tokens per word · {n}", "color": accent[n]} for n in results], "line", x_log=True)
cov = {}
for name, r in results.items():
    for c in r["top_k_coverage"]:
        cov.setdefault(c["k"], {"k": c["k"]})[f"cov_{name}"] = c["coverage"]
R.chart("topk", "Share of tokens covered by the k most frequent", [cov[k] for k in sorted(cov)], "k", [{"key": f"cov_{n}", "label": f"Coverage · {n}", "color": accent[n]} for n in results], "line", x_log=True, y_domain=[0, 1])
tpw = {}
for name, r in results.items():
    for b in r["tokens_per_word_hist"]:
        tpw.setdefault(b["bin"], {"bin": b["bin"]})[f"n_{name}"] = b["count"]
R.chart("tpw", "Tokens per word", list(tpw.values()), "bin", [{"key": f"n_{n}", "label": n, "color": accent[n]} for n in results], "bar")
tl = {}
for name, r in results.items():
    for b in r["token_length_hist"]:
        tl.setdefault(b["bin"], {"bin": b["bin"]})[f"n_{name}"] = b["count"]
R.chart("toklen", "Token lengths in use", list(tl.values()), "bin", [{"key": f"n_{n}", "label": n, "color": accent[n]} for n in results], "bar")

by_source = []
if holdout:
    from collections import defaultdict

    groups = defaultdict(list)
    for d in holdout:
        groups[d.get("source", "?")].append(d["text"])
    for src, texts in groups.items():
        r = bpe.evaluate(model, texts, curve_vocabs=[model.vocab_size])
        by_source.append({"source": src, "docs": len(texts), "chars_per_token": r["chars_per_token"], "tokens_per_word": r["tokens_per_word"], "char_coverage": r["char_coverage"], "whole_word_rate": r["whole_word_rate"]})
    R.table("by_source", "Held-out coverage by source", [{"key": "source", "label": "Source"}, {"key": "docs", "label": "Docs", "fmt": "int"}, {"key": "chars_per_token", "label": "Chars/token", "fmt": "num"}, {"key": "tokens_per_word", "label": "Tokens/word", "fmt": "num"}, {"key": "char_coverage", "label": "Coverage", "fmt": "pct"}, {"key": "whole_word_rate", "label": "Whole words", "fmt": "pct"}], by_source)
R.tokens("top", f"Most used tokens ({pname})", primary["top_tokens"])
if P.get("sample_text"):
    seg = model.segment(str(P["sample_text"]))
    R.tokens("sample", f"Your text in {len(seg)} tokens", seg, note="Colours follow the token kind: word, piece, number, space, mixed, bytes.")
R.output("tokenizer", I["tokenizer"]).output("chars_per_token", primary["chars_per_token"])
progress(100, "done")
R.save()
