"""Step 4 — called, parsed, used."""
import json
import os
import sys
from pathlib import Path

import torch

from atelier_sdk import Result, inputs, params, parse_args, progress
from atelier_mini.gen import generate
from atelier_mini.model import MiniLM
from atelier_mini.tok import MiniTokenizer
from atelier_world import World

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_tools import RESULT_CLOSE, RESULT_OPEN, calc, count, find_call, make_lookup, render, run_call  # noqa: E402

parse_args()
P = params({"n_eval": 300, "max_turns": 2, "max_new_tokens": 128})
I = inputs()
device = "cuda" if torch.cuda.is_available() else "cpu"
world = World(lang=I.get("lang", "en"), seed=33)
tok = MiniTokenizer.load(I["tokenizer"])
system = I.get("system") or world.system_prompt
tasks = world.eval_set(int(P["n_eval"]), seed=987_654_321)
TOOLS = {"calc": calc, "count": count, "lookup": make_lookup({})}


def run_loop(model, tasks, max_turns: int, label: str, offset: float):
    """Generate, run any tool call, feed the result back, repeat."""
    transcripts = [""] * len(tasks)
    stats = {"called": 0, "valid": 0, "invalid": 0, "used": 0}
    active = list(range(len(tasks)))
    for turn in range(max_turns + 1):
        if not active:
            break
        prompts = [tasks[i]["prompt"] + ("\n" + transcripts[i] if transcripts[i] else "") for i in active]
        gens = generate(model, tok, prompts, int(P["max_new_tokens"]), 0.0, batch_size=32, system=system, progress=lambda d, t: progress(offset + 20 * (turn + d / t) / (max_turns + 1), f"{label} turn {turn}: {d}/{t}"))
        still = []
        for idx, g in zip(active, gens):
            text = g[0]
            transcripts[idx] += text
            call = find_call(text, TOOLS)
            if call and turn < max_turns:
                name, argstr = call
                stats["called"] += 1
                result, ok = run_call(TOOLS, name, argstr)
                stats["valid" if ok else "invalid"] += 1
                transcripts[idx] += RESULT_OPEN + result + RESULT_CLOSE + "\n"
                still.append(idx)
        active = still
    return transcripts, stats


out = {}
for j, (name, path) in enumerate((("before", I["policy"]), ("after", I["tool_model"]))):
    model, _ = MiniLM.load(path, device)
    transcripts, stats = run_loop(model, tasks, int(P["max_turns"]), name, 5 + 45 * j)
    correct = [world.grade(t, task["answer"]) for t, task in zip(transcripts, tasks)]
    used = 0
    for t in transcripts:
        if RESULT_OPEN in t:
            result = t.split(RESULT_OPEN)[1].split(RESULT_CLOSE)[0]
            tail = t.split(RESULT_CLOSE, 1)[1] if RESULT_CLOSE in t else ""
            if result and result in tail:
                used += 1
    by_family = {}
    for c, task in zip(correct, tasks):
        f = by_family.setdefault(task["family"], [0, 0])
        f[0] += c
        f[1] += 1
    out[name] = {
        "accuracy": sum(correct) / len(tasks), "correct": correct, "transcripts": transcripts,
        "by_family": {k: v[0] / v[1] for k, v in by_family.items()},
        "call_rate": stats["called"] / len(tasks),
        "valid_rate": stats["valid"] / max(stats["called"], 1),
        "used_rate": used / max(sum(1 for t in transcripts if RESULT_OPEN in t), 1),
        "stats": stats,
    }
    del model
    torch.cuda.empty_cache()

b, a = out["before"], out["after"]
families = sorted(set(b["by_family"]) | set(a["by_family"]))
R = Result()
R.metric("accuracy", "Accuracy with tools", a["accuracy"], "pct", "kept")
R.metric("accuracy_before", "Accuracy before training on traces", b["accuracy"], "pct", "raw", help=f"the model could already call nothing: {b['call_rate']:.0%} call rate")
R.metric("call_rate", "Tool calls per question", a["call_rate"], "num", "sky")
R.metric("valid_calls", "Calls whose arguments parsed", a["valid_rate"], "pct", "kept" if a["valid_rate"] > 0.9 else "dup")
R.metric("used", "Results actually used in the answer", a["used_rate"], "pct", "hold", help="the failure that a final score hides: a correct call, then an answer that ignores it")
R.chart("accuracy", "Accuracy", [{"model": "before", "accuracy": b["accuracy"]}, {"model": "after", "accuracy": a["accuracy"]}], "model", [{"key": "accuracy", "label": "Correct", "color": "kept"}], "bar", y_domain=[0, 1])
R.chart("pipeline", "Where tool use breaks", [
    {"stage": "called a tool", "share": min(a["call_rate"], 1.0)},
    {"stage": "arguments parsed", "share": min(a["call_rate"], 1.0) * a["valid_rate"]},
    {"stage": "result used", "share": min(a["call_rate"], 1.0) * a["valid_rate"] * a["used_rate"]},
    {"stage": "answer correct", "share": a["accuracy"]},
], "stage", [{"key": "share", "label": "Share of questions", "color": "kept"}], "bar", y_domain=[0, 1], note="Each bar can only be as tall as the one before it. The largest drop is the thing to fix.")
R.chart("families", "Accuracy by family", [{"family": f, "before": b["by_family"].get(f, 0), "after": a["by_family"].get(f, 0)} for f in families], "family", [{"key": "before", "label": "Before", "color": "raw"}, {"key": "after", "label": "After", "color": "kept"}], "bar", y_domain=[0, 1])
R.table("transcripts", "Transcripts", [{"key": "ok", "label": ""}, {"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "transcript", "label": "What happened"}], [{"ok": "✓" if a["correct"][i] else "✗", "family": tasks[i]["family"], "question": tasks[i]["prompt"][:140], "transcript": a["transcripts"][i][:320]} for i in range(min(20, len(tasks)))])
R.output("accuracy", a["accuracy"]).output("valid_calls", a["valid_rate"]).output("tool_model", I["tool_model"]).output("tokenizer", I["tokenizer"]).output("lang", I.get("lang", "en"))
R.save()
