"""Step 1 — normalise instruction data into chat messages."""
import os
import random
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"dataset": "dolly", "max_examples": 4000, "max_chars": 3000, "val_frac": 0.05, "system_prompt": "", "seed": 1})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
rng = random.Random(int(P["seed"]))
NAMES = {"dolly": "dolly-15k", "alpaca": "alpaca-cleaned", "smoltalk": "smoltalk"}


def to_messages(row: dict) -> list[dict] | None:
    if "messages" in row:
        msgs = [{"role": m["role"], "content": m["content"]} for m in row["messages"] if m.get("role") in ("system", "user", "assistant")]
        return msgs if msgs and msgs[-1]["role"] == "assistant" else None
    if "instruction" in row:
        ctx = row.get("input") or row.get("context") or ""
        user = row["instruction"] + (f"\n\n{ctx}" if ctx else "")
        answer = row.get("output") or row.get("response") or ""
        return [{"role": "user", "content": user}, {"role": "assistant", "content": answer}] if answer.strip() else None
    if "prompt" in row and "completion" in row:
        return [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["completion"]}]
    return None


if P["dataset"] == "workspace":
    src = Path(I.get("workspace_dir", ".")) / "data" / "sft.jsonl"
    if not src.exists():
        raise SystemExit("Put a JSONL file at data/sft.jsonl in your project (rows with 'messages' or 'instruction'/'output').")
    rows = hf.read_jsonl(src)
else:
    rows = hf.dataset_split("sft", NAMES[P["dataset"]], "train")
rng.shuffle(rows)
progress(20, f"{len(rows)} raw rows")

examples, dropped_long, dropped_bad, cats = [], 0, 0, Counter()
for row in rows:
    msgs = to_messages(row)
    if not msgs:
        dropped_bad += 1
        continue
    if sum(len(m["content"]) for m in msgs) > int(P["max_chars"]):
        dropped_long += 1
        continue
    if P["system_prompt"].strip() and msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": P["system_prompt"].strip()}] + msgs
    examples.append({"messages": msgs, "category": row.get("category", "")})
    cats[row.get("category", "")] += 1
    if len(examples) >= int(P["max_examples"]):
        break
if len(examples) < 20:
    raise SystemExit(f"Only {len(examples)} usable examples; check the dataset format.")
n_val = max(10, int(len(examples) * float(P["val_frac"])))
val, train = examples[:n_val], examples[n_val:]
hf.write_jsonl(run_dir / "train.jsonl", train)
hf.write_jsonl(run_dir / "val.jsonl", val)
progress(80, "writing result")

prompt_lens = [sum(len(m["content"]) for m in e["messages"] if m["role"] != "assistant") for e in examples]
resp_lens = [sum(len(m["content"]) for m in e["messages"] if m["role"] == "assistant") for e in examples]
turns = Counter(sum(1 for m in e["messages"] if m["role"] == "assistant") for e in examples)
R = Result()
R.metric("train_examples", "Training examples", len(train), "int", "kept")
R.metric("val_examples", "Validation examples", len(val), "int", "hold")
R.metric("dropped", "Dropped", dropped_long + dropped_bad, "int", "dup", help=f"{dropped_long} too long, {dropped_bad} malformed")
R.metric("mean_response_chars", "Response length (chars)", sum(resp_lens) / len(resp_lens), "num", "sky")
R.chart("prompt_len", "Prompt length (characters)", hist(prompt_lens, bins=25, log=True), "bin", [{"key": "count", "label": "Examples", "color": "raw"}], "bar")
R.chart("resp_len", "Response length (characters)", hist(resp_lens, bins=25, log=True), "bin", [{"key": "count", "label": "Examples", "color": "kept"}], "bar")
R.chart("turns", "Assistant turns per example", [{"turns": str(k), "count": v} for k, v in sorted(turns.items())], "turns", [{"key": "count", "label": "Examples", "color": "sky"}], "bar")
if len(cats) > 1:
    R.chart("cats", "Categories", [{"category": k or "(none)", "count": v} for k, v in cats.most_common(12)], "category", [{"key": "count", "label": "Examples", "color": "hold"}], "bar")
R.table("samples", "Samples", [{"key": "user", "label": "User"}, {"key": "assistant", "label": "Assistant"}], [{"user": next((m["content"] for m in e["messages"] if m["role"] == "user"), "")[:300], "assistant": e["messages"][-1]["content"][:300]} for e in train[:20]])
R.artifact(run_dir / "train.jsonl", "train.jsonl").artifact(run_dir / "val.jsonl", "val.jsonl")
R.output("train", str(run_dir / "train.jsonl")).output("val", str(run_dir / "val.jsonl")).output("train_examples", len(train))
R.save()
