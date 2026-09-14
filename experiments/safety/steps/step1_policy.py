"""Step 1 — write the policy, then see how it divides real prompts."""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, params, parse_args, progress, write_jsonl
from atelier_nlp import hf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_safety import default_policy  # noqa: E402

parse_args()
P = params({"policy": "", "n_unsafe": 300, "n_helpful": 300, "seed": 1})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
rng = random.Random(int(P["seed"]))

unsafe, helpful, sources = [], [], []
try:
    pku = hf.dataset_split("safety", "pku-saferlhf", "train", limit=int(P["n_unsafe"]) * 6)
    rng.shuffle(pku)
    unsafe = [{"prompt": r["prompt"], "source": "pku-saferlhf"} for r in pku if r.get("prompt")][: int(P["n_unsafe"])]
    sources.append("pku-saferlhf")
except Exception as exc:
    progress(10, f"PKU-SafeRLHF not installed: {exc}")
try:
    tox = hf.dataset_split("safety", "toxicity-prompts", "train", limit=int(P["n_unsafe"]) * 4)
    rng.shuffle(tox)
    extra = [{"prompt": (r.get("prompt") or {}).get("text") if isinstance(r.get("prompt"), dict) else r.get("prompt"), "source": "toxicity-prompts"} for r in tox if r.get("challenging")]
    unsafe += [e for e in extra if e["prompt"]][: int(P["n_unsafe"]) // 2]
    sources.append("toxicity-prompts")
except Exception as exc:
    progress(20, f"RealToxicityPrompts not installed: {exc}")
try:
    rows = hf.dataset_split("sft", "no-robots", "train", limit=int(P["n_helpful"]) * 4)
    rng.shuffle(rows)
    for r in rows:
        msgs = r.get("messages") or []
        user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
        if user:
            helpful.append({"prompt": user, "source": "no-robots", "category_hint": r.get("category", "")})
        if len(helpful) >= int(P["n_helpful"]):
            break
    sources.append("no-robots")
except Exception as exc:
    progress(30, f"No Robots not installed: {exc}")
if not unsafe and not helpful:
    raise SystemExit("None of the safety datasets are installed. Fetch the safety and sft groups first.")

for r in unsafe:
    r["policy"] = default_policy(r["prompt"])
    r["expected"] = "refuse"
for r in helpful:
    r["policy"] = default_policy(r["prompt"])
    r["expected"] = "normal"
progress(60, "classifying")

all_rows = unsafe + helpful
write_jsonl(run_dir / "prompts.jsonl", all_rows)
(run_dir / "policy.json").write_text(json.dumps({"policy": P["policy"], "sources": sources}, ensure_ascii=False, indent=2))
agree_unsafe = sum(1 for r in unsafe if r["policy"] == "refuse") / max(len(unsafe), 1)
agree_helpful = sum(1 for r in helpful if r["policy"] == "normal") / max(len(helpful), 1)
cats = Counter(r["policy"] for r in all_rows)

R = Result()
R.metric("prompts", "Prompts loaded", len(all_rows), "int", "kept", help=f"{len(unsafe)} from safety sets, {len(helpful)} ordinary requests")
R.metric("policy_catches", "Unsafe prompts the keyword policy catches", agree_unsafe, "pct", "dup" if agree_unsafe < 0.5 else "hold", help="the starting policy is crude on purpose")
R.metric("policy_passes", "Ordinary requests it leaves alone", agree_helpful, "pct", "kept")
R.metric("categories", "Categories in use", len(cats), "int", "sky")
R.chart("categories", "How the policy divides the prompts", [{"category": c, "unsafe": sum(1 for r in unsafe if r["policy"] == c), "ordinary": sum(1 for r in helpful if r["policy"] == c)} for c in ("refuse", "care", "normal")], "category", [{"key": "unsafe", "label": "From the safety sets", "color": "dup"}, {"key": "ordinary", "label": "Ordinary requests", "color": "kept"}], "bar", note="Every ordinary request in the 'refuse' column is a request the model will learn to decline. Every safety prompt in 'normal' is one it will answer.")
R.table("policy", "Your policy", [{"key": "line", "label": "Rule"}], [{"line": l} for l in str(P["policy"]).splitlines() if l.strip()])
R.table("edge", "Where the keyword policy disagrees with the source", [{"key": "expected", "label": "Source says"}, {"key": "policy", "label": "Policy says"}, {"key": "prompt", "label": "Prompt"}], [{"expected": r["expected"], "policy": r["policy"], "prompt": r["prompt"][:220]} for r in all_rows if r["policy"] != r["expected"]][:25], note="These are the cases your policy has to decide. Reading twenty of them is worth more than another rule.")
R.artifact(run_dir / "prompts.jsonl", "prompts.jsonl")
R.output("prompts", str(run_dir / "prompts.jsonl")).output("policy_file", str(run_dir / "policy.json")).output("n_unsafe", len(unsafe)).output("n_helpful", len(helpful))
R.save()
