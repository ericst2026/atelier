"""Step 1 — build the benchmark (generated from the world, or from a prepared QA dataset) and look at it before scoring anything."""
import os
import random
import sys
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, params, parse_args, progress, write_jsonl
from atelier_world import World, multiple_choice
from atelier_world.prepared import dataset_splits, read_qa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib_items import render_item  # noqa: E402

parse_args()
P = params({"data_source": "generated", "data_material": None, "n_choice": 600, "n_open": 300, "n_options": 4, "families": ["arith", "count", "sort", "lookup", "path", "shop", "compare", "seq"], "lang": "en", "seed": 4242})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
prepared = P["data_source"] == "prepared"
fams = None if prepared else (list(P["families"]) or None)
world = World(lang=P["lang"], seed=int(P["seed"]), families=fams)
rng = random.Random(int(P["seed"]))
n_opt = int(P["n_options"])


def _norm(a: str) -> str:
    return " ".join(str(a).split()).lower()


def prepared_choice(item: dict, by_family: dict, pool: list, rng: random.Random) -> dict:
    """A multiple-choice item from a prepared question. There is no generator to draw
    distractors from, so they are other rows' answers: the same family (category /
    subject) first, numbers for a numeric answer, then anything else in the dataset."""
    gold = str(item["answer"])
    numeric = gold.lstrip("-").replace(".", "", 1).isdigit()
    same = [a for a in by_family.get(item["family"], []) if _norm(a) != _norm(gold)]
    rest = [a for a in pool if _norm(a) != _norm(gold)]
    rng.shuffle(same)
    rng.shuffle(rest)
    cands = same + rest
    if numeric:
        cands = sorted(cands, key=lambda a: not a.lstrip("-").replace(".", "", 1).isdigit())  # stable: numbers first
    distractors, seen = [], {_norm(gold)}
    for a in cands:
        if _norm(a) not in seen:
            seen.add(_norm(a))
            distractors.append(a)
        if len(distractors) == n_opt - 1:
            break
    if gold.lstrip("-").isdigit() and len(distractors) >= 2:
        near = str(int(gold) + rng.choice([-10, -2, -1, 1, 2, 10]))
        if _norm(near) not in seen:
            distractors[-1] = near
    options = distractors + [gold]
    rng.shuffle(options)
    return {"question": item["prompt"], "options": options, "answer": options.index(gold), "family": item["family"], "difficulty": item.get("difficulty", 1), "gold_text": gold}


if prepared:
    rel = P["data_material"]
    if not rel:
        raise SystemExit("Choose a prepared QA dataset from the list, or switch the benchmark back to generated.")
    splits = set(dataset_splits(rel))
    split = next((s for s in ("test", "validation", "val", "dev") if s in splits), None)
    progress(10, f"reading materials/{rel}" + (f" ({split})" if split else ""))
    rows = read_qa(rel, split=split)
    rng.shuffle(rows)
    want_c, want_o = int(P["n_choice"]), int(P["n_open"])
    if len(rows) < want_c + want_o:
        # not enough rows for both: share them in proportion, never the same row twice
        want_c = round(len(rows) * want_c / max(want_c + want_o, 1)) if want_o else len(rows)
        want_o = len(rows) - want_c if want_o else 0
    pool = sorted({str(r["answer"]) for r in rows})
    if len({_norm(a) for a in pool}) < n_opt:
        raise SystemExit(f"materials/{rel} has only {len(pool)} different answers, too few to make {n_opt} options per item: distractors for a prepared dataset are drawn from the other rows' answers. Lower the options per item, or use a dataset with more varied answers.")
    by_family: dict = {}
    for r in rows:
        by_family.setdefault(r["family"], set()).add(str(r["answer"]))
    by_family = {f: sorted(v) for f, v in by_family.items()}
    choice = [prepared_choice(it, by_family, pool, rng) for it in rows[:want_c]]
    open_items = [{"question": it["prompt"], "answer": it["answer"], "family": it["family"], "difficulty": it["difficulty"]} for it in rows[want_c : want_c + want_o]]
    data_label = f"materials/{rel}" + (f" ({split})" if split else "")
else:
    base = world.eval_set(int(P["n_choice"]), seed=int(P["seed"]) * 7 + 1, families=fams)
    choice = [multiple_choice(world, it, rng, n_opt) for it in base]
    open_items = [{"question": it["prompt"], "answer": it["answer"], "family": it["family"], "difficulty": it["difficulty"]} for it in world.eval_set(int(P["n_open"]), seed=int(P["seed"]) * 13 + 5, families=fams)] if int(P["n_open"]) else []
    data_label = "generated from the world"
if not choice:
    raise SystemExit("No multiple-choice items: raise the number of multiple-choice items.")
progress(45, f"{len(choice)} multiple-choice items")
write_jsonl(run_dir / "choice.jsonl", choice)
write_jsonl(run_dir / "open.jsonl", open_items)
progress(75, "checking the items for cheap giveaways")

longest = sum(1 for x in choice if max(range(len(x["options"])), key=lambda j: len(x["options"][j])) == x["answer"]) / max(len(choice), 1)
first = sum(1 for x in choice if x["answer"] == 0) / max(len(choice), 1)
dup_options = sum(1 for x in choice if len(set(x["options"])) < len(x["options"])) / max(len(choice), 1)
fam_counts = Counter(x["family"] for x in choice)
by_family = [{"family": f, "count": c, "longest_correct": sum(1 for x in choice if x["family"] == f and max(range(len(x["options"])), key=lambda j: len(x["options"][j])) == x["answer"]) / c} for f, c in sorted(fam_counts.items())]

R = Result()
R.metric("items", "Items", len(choice) + len(open_items), "int", "kept", help=f"{len(choice)} multiple choice, {len(open_items)} open · {data_label}")
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
R.output("data_source", P["data_source"]).output("data_label", data_label)
if prepared:
    R.note(f"The items come from {data_label}. Its rows have no generator behind them, so each item's wrong options are other rows' answers — the same category first, numbers for a numeric answer. That makes them the same kind of thing, but not always the same size: check the longest-option bias above before trusting the scores.")
R.save()
