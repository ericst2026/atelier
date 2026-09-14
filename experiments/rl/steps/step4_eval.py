"""Step 4 — before/after accuracy, lengths, and reward-hacking flags."""
import json
import os
import re
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"n_eval": 300, "max_new_tokens": 384})
I = inputs()
spec = json.loads(Path(I["reward_spec"]).read_text())
test = hf.read_jsonl(I["test_prompts"], limit=int(P["n_eval"]))


def flags(text: str) -> list[str]:
    f = []
    if hf.extract_answer(text) is None:
        f.append("no answer")
    if len(re.findall(r"####", text)) > 1:
        f.append("multiple answers")
    if len(text) > 2500:
        f.append("very long")
    words = text.split()
    if len(words) > 30 and len(set(words)) / len(words) < 0.3:
        f.append("repetitive")
    return f


out = {}
for j, (name, path) in enumerate((("before", I["init_policy"]), ("after", I["policy_dir"]))):
    tok = hf.load_tokenizer(path)
    model = hf.load_model(path)
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in test]
    gens = hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), 0.0, progress=lambda d, t: progress(5 + 45 * j + 40 * d / t, f"{name}: {d}/{t}"))
    answers = [g[0] for g in gens]
    correct = [hf.answers_equal(hf.extract_answer(a), r["answer"]) for a, r in zip(answers, test)]
    marked = [bool(re.search(r"####|\\boxed|answer is", a, re.I)) for a in answers]
    out[name] = {"answers": answers, "correct": correct, "acc": sum(correct) / len(test), "marked": sum(marked) / len(test), "lens": [len(a) for a in answers], "flags": [flags(a) for a in answers]}
    del model
    import torch

    torch.cuda.empty_cache()

b, a = out["before"], out["after"]
flag_counts = {}
for name in ("before", "after"):
    for fl in out[name]["flags"]:
        for f in fl:
            flag_counts.setdefault(f, {"flag": f, "before": 0, "after": 0})[name] += 1
R = Result()
R.metric("acc_before", "Accuracy before", b["acc"], "pct", "raw")
R.metric("acc_after", "Accuracy after", a["acc"], "pct", "kept", help=f"{sum(a['correct'])} of {len(test)}")
R.metric("marked_after", "Marked final answer", a["marked"], "pct", "sky", help=f"before: {100 * b['marked']:.0f}%")
R.metric("len_after", "Answer length (chars)", sum(a["lens"]) / len(test), "num", "hold", help=f"before: {sum(b['lens']) / len(test):.0f}")
R.chart("acc", "Accuracy on held-out problems", [{"policy": "before", "accuracy": b["acc"]}, {"policy": "after", "accuracy": a["acc"]}], "policy", [{"key": "accuracy", "label": "Accuracy", "color": "kept"}], "bar", y_domain=[0, 1])
lb = {x["bin"]: x for x in hist(b["lens"], bins=15, lo=0, hi=max(max(b["lens"]), max(a["lens"])))}
la = hist(a["lens"], bins=15, lo=0, hi=max(max(b["lens"]), max(a["lens"])))
R.chart("lens", "Answer length distribution", [{"bin": x["bin"], "before": lb.get(x["bin"], {}).get("count", 0), "after": x["count"]} for x in la], "bin", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar")
if flag_counts:
    R.chart("flags", "Suspicious completions", list(flag_counts.values()), "flag", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "dup"}], "bar", note="Reward hacking looks like more marked answers with no more correct ones, or growth in repetitive/very long completions.")
changed = [i for i in range(len(test)) if b["correct"][i] != a["correct"][i]]
R.table("changed", "Problems whose outcome changed", [{"key": "question", "label": "Question"}, {"key": "gold", "label": "Gold"}, {"key": "before", "label": "Before"}, {"key": "after", "label": "After"}, {"key": "flags", "label": "Flags"}], [{"question": test[i]["question"][:300], "gold": test[i]["answer"], "before": ("✓ " if b["correct"][i] else "✗ ") + b["answers"][i][:400], "after": ("✓ " if a["correct"][i] else "✗ ") + a["answers"][i][:400], "flags": ", ".join(a["flags"][i])} for i in changed[:40]])
R.output("accuracy", a["acc"]).output("accuracy_before", b["acc"])
R.save()
