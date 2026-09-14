"""Step 3 — rejection-sampled fine-tuning (STaR-style)."""
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"model": "Qwen2.5-0.5B-Instruct", "k": 8, "temperature": 0.8, "max_keep": 2, "max_new_tokens": 384, "epochs": 2.0, "lr": 1e-4, "lora_r": 16})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
train = hf.read_jsonl(I["train"])
mp = hf.model_path(P["model"])
tok = hf.load_tokenizer(mp)
model = hf.load_model(mp)
COT = "Solve the problem step by step, then give the final answer on its own line as '#### <number>'."
k = int(P["k"])
prompts = [hf.chat_prompt(tok, e["question"], COT) for e in train]
gens = hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), float(P["temperature"]), batch_size=8, num_return_sequences=k, progress=lambda d, t: progress(60 * d / t, f"sampling {d}/{t} problems × {k}"))
del model
import torch  # noqa: E402

torch.cuda.empty_cache()

kept_rows, per_problem, coverage = [], [], [0] * k
for e, gs in zip(train, gens):
    good = [g for g in gs if hf.answers_equal(hf.extract_answer(g), e["gold"])]
    per_problem.append(len(good))
    first = next((i for i, g in enumerate(gs) if hf.answers_equal(hf.extract_answer(g), e["gold"])), None)
    if first is not None:
        for j in range(first, k):
            coverage[j] += 1
    good.sort(key=len)
    for g in good[: int(P["max_keep"])]:
        kept_rows.append({"messages": [{"role": "system", "content": COT}, {"role": "user", "content": e["question"]}, {"role": "assistant", "content": g.strip()}]})
hf.write_jsonl(run_dir / "rft_train.jsonl", kept_rows)
if len(kept_rows) < 10:
    raise SystemExit(f"Only {len(kept_rows)} correct samples — raise k or temperature, or use a stronger model.")
progress(62, f"{len(kept_rows)} correct solutions kept from {sum(1 for n in per_problem if n)} solvable problems")
res = hf.sft_train(mp, kept_rows, run_dir, None, method="lora", lora={"r": int(P["lora_r"]), "alpha": 2 * int(P["lora_r"])}, epochs=float(P["epochs"]), lr=float(P["lr"]), batch_size=4, grad_accum=4, max_length=1024, label="rft")
R = Result()
R.metric("kept", "Solutions kept", len(kept_rows), "int", "kept")
R.metric("solvable", "Problems solved at least once", sum(1 for n in per_problem if n) / len(train), "pct", "sky", help=f"of {len(train)} with k={k}")
R.metric("train_time", "Training time", res["elapsed_sec"] * 1000, "ms", "hold")
R.chart("per_problem", "Correct samples per problem", [{"bin": str(i), "count": c} for i, c in sorted(Counter(per_problem).items())], "bin", [{"key": "count", "label": "Problems", "color": "kept"}], "bar")
R.chart("coverage", "Problems solved at least once vs samples", [{"k": j + 1, "coverage": coverage[j] / len(train)} for j in range(k)], "k", [{"key": "coverage", "label": "Coverage", "color": "sky"}], "line", y_domain=[0, 1])
R.chart("loss", "Fine-tuning loss", hf.history_chart(res["history"], ["loss"]), "step", [{"key": "loss", "label": "Loss", "color": "hold"}], "line")
R.chart("lens", "Kept solution lengths (characters)", hist([len(r["messages"][-1]["content"]) for r in kept_rows], bins=20), "bin", [{"key": "count", "label": "Solutions", "color": "raw"}], "bar")
R.artifact(run_dir / "rft_train.jsonl", "rft_train.jsonl").artifact(Path(res["save_dir"]), "LoRA adapter")
R.output("adapter", res["save_dir"]).output("base_model", str(mp)).output("eval", I["eval"]).output("kept", len(kept_rows))
R.save()
