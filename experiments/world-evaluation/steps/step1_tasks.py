"""Step 1 — build the benchmark and look at it before scoring anything."""
import os
import random
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, params, parse_args, progress, write_jsonl
from atelier_world import World, multiple_choice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_items import render_item  # noqa: E402

parse_args()
P = params({"n_choice": 600, "n_open": 300, "n_options": 4, "families": ["arith", "count", "sort", "lookup", "path", "shop", "compare", "seq"], "lang": "en", "seed": 4242})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
fams = list(P["families"]) or None
world = World(lang=P["lang"], seed=int(P["seed"]), families=fams)
rng = random.Random(int(P["seed"]))

base = world.eval_set(int(P["n_choice"]), seed=int(P["seed"]) * 7 + 1, families=fams)
choice = [multiple_choice(world, it, rng, int(P["n_options"])) for it in base]
progress(45, f"{len(choice)} multiple-choice items")
open_items = [{"question": it["prompt"], "answer": it["answer"], "family": it["family"], "difficulty": it["difficulty"]} for it in world.eval_set(int(P["n_open"]), seed=int(P["seed"]) * 13 + 5, families=fams)] if int(P["n_open"]) else []
write_jsonl(run_dir / "choice.jsonl", choice)
write_jsonl(run_dir / "open.jsonl", open_items)
progress(75, "checking the items for cheap giveaways")

n_opt = int(P["n_options"])
longest = sum(1 for x in choice if max(range(len(x["options"])), key=lambda j: len(x["options"][j])) == x["answer"]) / max(len(choice), 1)
first = sum(1 for x in choice if x["answer"] == 0) / max(len(choice), 1)
dup_options = sum(1 for x in choice if len(set(x["options"])) < len(x["options"])) / max(len(choice), 1)
fam_counts = Counter(x["family"] for x in choice)
by_family = [{"family": f, "count": c, "longest_correct": sum(1 for x in choice if x["family"] == f and max(range(len(x["options"])), key=lambda j: len(x["options"][j])) == x["answer"]) / c} for f, c in sorted(fam_counts.items())]

R = Result()
R.metric("items", "Items", len(choice) + len(open_items), "int", "kept", help=f"{len(choice)} multiple choice, {len(open_items)} open")
R.metric("chance", "Chance", 1 / n_opt, "pct", "hold", help=f"{n_opt} options per item")
R.metric("longest_correct", "Longest option is correct", longest, "pct", "dup" if longest > 1.4 / n_opt else "kept", help="a model that prefers long answers scores above chance without understanding anything")
R.metric("duplicate_options", "Items with a repeated option", dup_options, "pct", "dup" if dup_options > 0.02 else "kept")
R.chart("families", "Items by family", [{"family": f, "count": c} for f, c in sorted(fam_counts.items())], "family", [{"key": "count", "label": "Items", "color": "kept"}], "bar")
R.chart("bias", "Where the longest option gives it away", [{"family": r["family"], "longest": r["longest_correct"], "chance": 1 / n_opt} for r in by_family], "family", [{"key": "longest", "label": "Longest option correct", "color": "dup"}, {"key": "chance", "label": "Chance", "color": "hold"}], "bar", y_domain=[0, 1], note="Where the first bar clears the second, the item leaks its answer through length. This is exactly why length normalisation is a real decision in the next step and not a detail.")
R.chart("position", "Answer position", [{"position": str(i), "share": sum(1 for x in choice if x["answer"] == i) / max(len(choice), 1)} for i in range(n_opt)], "position", [{"key": "share", "label": "Share", "color": "sky"}], "bar", y_domain=[0, 1], note=f"Should be flat at {1 / n_opt:.0%}. A model that always picks the first option would otherwise score well.")
R.table("items", "What the items look like", [{"key": "family", "label": "Family"}, {"key": "question", "label": "Question"}, {"key": "options", "label": "Options"}, {"key": "answer", "label": "Correct"}], [{"family": x["family"], "question": x["question"][:180], "options": " | ".join(x["options"]), "answer": x["options"][x["answer"]]} for x in choice[:15]])
R.table("rendered", "How a leaked item would appear in a training corpus", [{"key": "text", "label": "Text"}], [{"text": render_item(x)} for x in choice[:3]], note="The last step mixes exactly this text into pretraining, to measure what contamination is worth.")
R.artifact(run_dir / "choice.jsonl", "choice.jsonl").artifact(run_dir / "open.jsonl", "open.jsonl")
R.output("choice", str(run_dir / "choice.jsonl")).output("open", str(run_dir / "open.jsonl")).output("lang", P["lang"]).output("seed", int(P["seed"])).output("n_options", n_opt).output("families", fams)
R.save()
