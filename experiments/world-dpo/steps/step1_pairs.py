"""Step 1 — build preference pairs from the model's own attempts."""
import os
from collections import Counter
from pathlib import Path

import torch

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

parse_args()
P = params({"n_questions": 3000, "k": 6, "temperature": 1.0, "pair_on": "correct", "max_pairs_per_question": 1, "max_new_tokens": 160})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("sft_run_run")
if not ref:
    raise SystemExit("Choose a Fine-tuning run (step 3).")
o = ref["outputs"]
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=o.get("lang", "en"), seed=6)
model, ck = MiniLM.load(o["sft_model"], device)
tok = MiniTokenizer.load(o["tokenizer"])
system = o.get("system") or ck.get("system") or world.system_prompt

n, k = int(P["n_questions"]), int(P["k"])
tasks = world.eval_set(n, seed=707_001)
progress(3, f"sampling {k} answers for each of {n} questions")
groups = generate(model, tok, [t["prompt"] for t in tasks], int(P["max_new_tokens"]), float(P["temperature"]), num_samples=k, batch_size=max(16, k * 2), system=system, progress=lambda d, t: progress(5 + 80 * d / t, f"{d}/{t}"))

pairs, reasons, no_pair = [], Counter(), 0
for t, group in zip(tasks, groups):
    right = [c for c in group if world.grade(c, t["answer"])]
    wrong = [c for c in group if not world.grade(c, t["answer"])]
    made = 0
    limit = int(P["max_pairs_per_question"])
    if P["pair_on"] in ("correct", "mixed") and right and wrong:
        for c, r in zip(sorted(right, key=len), sorted(wrong, key=len, reverse=True)):
            if made >= limit:
                break
            pairs.append({"prompt": t["prompt"], "chosen": c.strip(), "rejected": r.strip(), "family": t["family"], "basis": "correct"})
            reasons["correct"] += 1
            made += 1
    if P["pair_on"] in ("correct_short", "mixed") and len(right) >= 2 and made < limit:
        s = sorted(right, key=len)
        if len(s[-1]) > len(s[0]) * 1.3:
            pairs.append({"prompt": t["prompt"], "chosen": s[0].strip(), "rejected": s[-1].strip(), "family": t["family"], "basis": "shorter"})
            reasons["shorter"] += 1
            made += 1
    if made == 0:
        no_pair += 1

if len(pairs) < 20:
    raise SystemExit(f"Only {len(pairs)} pairs. Raise the temperature or the number of answers per question — the model is answering too consistently to disagree with itself.")
split = max(20, len(pairs) // 10)
write_jsonl(run_dir / "pairs.jsonl", pairs[split:])
write_jsonl(run_dir / "pairs_val.jsonl", pairs[:split])
fam = Counter(p["family"] for p in pairs)
len_chosen = [len(p["chosen"]) for p in pairs]
len_rejected = [len(p["rejected"]) for p in pairs]

R = Result()
R.metric("pairs", "Pairs built", len(pairs), "int", "kept", help=f"{no_pair} questions produced none")
R.metric("coverage", "Questions that produced a pair", 1 - no_pair / len(tasks), "pct", "sky", help="too easy or too hard to disagree with itself")
R.metric("chosen_len", "Chosen answer length", sum(len_chosen) / len(pairs), "num", "hold", help=f"rejected: {sum(len_rejected) / len(pairs):.0f} characters")
R.chart("basis", "What made one answer better", [{"basis": b, "count": c} for b, c in reasons.items()], "basis", [{"key": "count", "label": "Pairs", "color": "kept"}], "bar")
R.chart("families", "Pairs by family", [{"family": f, "count": c} for f, c in sorted(fam.items())], "family", [{"key": "count", "label": "Pairs", "color": "raw"}], "bar", note="Families missing here cannot be improved by this data — the model never disagreed with itself on them.")
R.chart("lengths", "Length of chosen answers", hist(len_chosen, bins=16), "bin", [{"key": "count", "label": "Pairs", "color": "hold"}], "bar", note="If chosen answers are systematically shorter, the model will learn brevity along with correctness — sometimes that is what you wanted, sometimes not.")
R.table("pairs", "A sample", [{"key": "family", "label": "Family"}, {"key": "prompt", "label": "Question"}, {"key": "chosen", "label": "Chosen"}, {"key": "rejected", "label": "Rejected"}], [{"family": p["family"], "prompt": p["prompt"][:160], "chosen": p["chosen"][:200], "rejected": p["rejected"][:200]} for p in pairs[:20]])
R.artifact(run_dir / "pairs.jsonl", "pairs.jsonl")
R.output("pairs", str(run_dir / "pairs.jsonl")).output("pairs_val", str(run_dir / "pairs_val.jsonl")).output("policy", o["sft_model"]).output("tokenizer", o["tokenizer"]).output("lang", o.get("lang", "en")).output("system", system).output("sft_accuracy", o.get("accuracy"))
R.save()
