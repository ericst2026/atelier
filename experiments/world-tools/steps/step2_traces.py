"""Step 2 — build demonstrations of calling a tool and using what comes back."""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, read_jsonl, write_jsonl
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_tools import calc, count, prepared_call, render  # noqa: E402

parse_args()
P = params({"n_traces": 8000, "no_tool_share": 0.25, "seed": 33})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
cfg = json.loads(Path(I["tools_config"]).read_text())
enabled = set(cfg["enabled"])
world = World(lang=I.get("lang", "en"), seed=int(P["seed"]))
rng = random.Random(int(P["seed"]))
n = int(P["n_traces"])
prefix = world.answer_prefix

rows, skipped, by_tool = [], 0, Counter()
prepared = I.get("data_source") == "prepared"
if prepared:
    tasks = read_jsonl(I["qa_train"])
    rng.shuffle(tasks)
else:
    tasks = world.eval_set(n * 2, seed=int(P["seed"]) * 7 + 1)
for t in tasks:
    if len(rows) >= n:
        break
    fam, answer = t["family"], t["answer"]
    want_no_tool = rng.random() < float(P["no_tool_share"])
    target = None
    if prepared:
        # a prepared question: call a tool only where its result is the dataset's answer;
        # every other question becomes a demonstration of answering without one
        call = None if want_no_tool else prepared_call(t, enabled, lambda r: world.grade(f"{prefix} {r}", answer))
        if call:
            target = render(*call) + f"\n{prefix} {answer}"
            by_tool[call[0]] += 1
        else:
            target = "".join(f"{line}\n" for line in t["steps"]) + f"{prefix} {answer}"
            by_tool["none"] += 1
    elif want_no_tool or fam in ("sort", "lookup", "path"):
        target = "\n".join(t["steps"]) + f"\n{prefix} {answer}"
        by_tool["none"] += 1
    elif "calc" in enabled and fam in ("arith", "shop", "compare", "seq"):
        # find the arithmetic in the reasoning steps and turn it into a call
        expr = None
        for line in t["steps"]:
            m = [s for s in line.replace("=", " = ").split() if s]
            if "=" in line:
                left = line.split("=")[0]
                candidate = "".join(ch for ch in left if ch.isdigit() or ch in "+-*/(). ").strip()
                if candidate and any(ch.isdigit() for ch in candidate) and any(op in candidate for op in "+-*/"):
                    expr = candidate
        if expr:
            try:
                result = calc(expr)
            except Exception:
                result = None
            if result is not None:
                head = "\n".join(t["steps"][:-2]) if len(t["steps"]) > 2 else ""
                target = (head + "\n" if head else "") + render("calc", expr, result) + f"\n{prefix} {answer}"
                by_tool["calc"] += 1
    elif "count" in enabled and fam == "count":
        numbers = [x for x in t["steps"][0].replace(",", " ").replace("、", " ").split() if x.strip(".,").isdigit()]
        if numbers:
            args = ", ".join(numbers)
            try:
                result = count(args)
            except Exception:
                result = None
            if result is not None and result == answer:
                target = render("count", args, result) + f"\n{prefix} {answer}"
                by_tool["count"] += 1
    if target is None:
        skipped += 1
        continue
    rows.append({"prompt": t["prompt"], "target": target, "family": fam, "answer": answer, "tool": "none" if "(" not in target.split("\n")[0] else target.split("(")[0].split("\n")[-1]})

if len(rows) < 50:
    raise SystemExit(f"Only {len(rows)} usable demonstrations. " + ("The prepared dataset needs more training rows (at least 100: 50 are held out for validation)." if prepared else "Enable more tools, or lower the share showing no tool call."))
split = max(50, len(rows) // 10)
write_jsonl(run_dir / "traces.jsonl", rows[split:])
write_jsonl(run_dir / "traces_val.jsonl", rows[:split])
target_lens = [len(r["target"]) for r in rows]
fam_counts = Counter(r["family"] for r in rows)

R = Result()
R.metric("traces", "Demonstrations", len(rows), "int", "kept", help=f"{skipped} questions produced none")
R.metric("with_tool", "Showing a tool call", 1 - by_tool["none"] / max(len(rows), 1), "pct", "sky")
R.metric("tools_used", "Distinct tools demonstrated", len([k for k in by_tool if k != "none"]), "int", "hold")
R.metric("target_chars", "Target length", sum(target_lens) / len(rows), "num", "raw", help="shorter than a full chain of working, because the tool does the arithmetic")
R.chart("tools", "Demonstrations by tool", [{"tool": k, "count": v} for k, v in by_tool.most_common()], "tool", [{"key": "count", "label": "Demonstrations", "color": "kept"}], "bar", note="The 'none' bar matters. Without it the model learns that every question deserves a call, including the ones a tool cannot help with.")
R.chart("families", "By family", [{"family": k, "count": v} for k, v in sorted(fam_counts.items())], "family", [{"key": "count", "label": "Demonstrations", "color": "sky"}], "bar")
R.chart("lengths", "Target lengths", hist(target_lens, bins=18), "bin", [{"key": "count", "label": "Demonstrations", "color": "hold"}], "bar")
R.table("samples", "What the model will learn to write", [{"key": "family", "label": "Family"}, {"key": "prompt", "label": "Question"}, {"key": "target", "label": "Target"}], [{"family": r["family"], "prompt": r["prompt"][:150], "target": r["target"][:260]} for r in rows[:20]], note="The bracketed result is what the tool returned. In training it sits in the context but is masked out of the loss — the model is not learning to predict the calculator.")
R.artifact(run_dir / "traces.jsonl", "traces.jsonl")
R.output("traces", str(run_dir / "traces.jsonl")).output("traces_val", str(run_dir / "traces_val.jsonl"))
for k in ("tools_config", "policy", "tokenizer", "lang", "system", "baseline", "model_format", "adapter", "model_label", "data_source", "qa_val"):
    if k in I:
        R.output(k, I[k])
if prepared:
    R.note("Traces from the prepared dataset's training rows. A call is demonstrated only where the tool's result is the dataset's answer; the seed only shuffles the rows. Held-out rows are kept for step 4.")
R.save()
