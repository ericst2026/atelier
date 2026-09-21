"""Step 1 — demonstrations: generated from the world, or a prepared dataset."""
import json
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_world import World
from atelier_world.prepared import choose_model, model_outputs, read_qa, train_val

parse_args()
P = params({"model_source": "generated", "data_source": "generated", "examples": 20000, "with_steps": True, "families": ["arith", "count", "sort", "lookup", "path", "shop", "compare", "seq"], "difficulty": "mixed", "val_examples": 500, "seed": 21})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
base = choose_model(P, I, run_key="base_run", run_model_key="model", hint="Choose a Pretraining run (step 3), or a prepared model — this experiment fine-tunes a base model.")
lang = base["lang"]
with_steps = bool(P["with_steps"])
prepared = P["data_source"] == "prepared"
families = None if prepared else (list(P["families"]) or None)
difficulty = None if prepared or P["difficulty"] == "mixed" else int(P["difficulty"])
world = World(lang=lang, seed=int(P["seed"]), families=families)


def target(steps: list, answer: str) -> str:
    body = ("\n".join(steps) + "\n") if with_steps and steps else ""
    return f"{body}{world.answer_prefix} {answer}"


n = int(P["examples"])
if prepared:
    progress(10, f"reading materials/{P['data_material']}")
    train_qa, val_qa = train_val(P["data_material"], read_qa, int(P["val_examples"]), seed=int(P["seed"]), limit=n)
    rows = [{"id": i, "family": q["family"], "difficulty": q["difficulty"], "prompt": q["prompt"], "target": target(q["steps"], q["answer"]), "answer": q["answer"], "steps": q["steps"]} for i, q in enumerate(train_qa[:n])]
    val = val_qa
    data_label = f"materials/{P['data_material']}"
else:
    rows = []
    by_family: dict = {}
    for i, ex in enumerate(world.instructions(n, with_steps=with_steps, difficulty=difficulty)):
        rows.append(ex)
        by_family[ex.get("family", "other")] = by_family.get(ex.get("family", "other"), 0) + 1
        if i % 2000 == 0:
            progress(70 * i / n, f"{i:,} demonstrations", step=i + 1, x_label="demonstrations written", **{f"demonstrations__{k}": v for k, v in by_family.items()})
    val = world.eval_set(int(P["val_examples"]), seed=555_003, difficulty=difficulty, families=families)
    data_label = "generated from the world"
# the held-out questions steps 2 and 4 grade, the same for both sources
val_rows = [{"prompt": v["prompt"], "target": target(v["steps"], v["answer"]), "answer": v["answer"], "family": v["family"], "difficulty": v["difficulty"], "steps": v["steps"]} for v in val]
write_jsonl(run_dir / "train.jsonl", rows)
write_jsonl(run_dir / "val.jsonl", val_rows)
(run_dir / "sft_meta.json").write_text(json.dumps({"lang": lang, "with_steps": with_steps, "families": families, "system": world.system_prompt, "data_source": P["data_source"], "data_label": data_label, "model_label": base["label"]}, ensure_ascii=False, indent=2))
progress(85, "writing result")

fam = Counter(r["family"] for r in rows)
diff = Counter(r["difficulty"] for r in rows)
target_lens = [len(r["target"]) for r in rows]
prompt_lens = [len(r["prompt"]) for r in rows]

R = Result()
R.metric("examples", "Demonstrations", len(rows), "int", "kept", help=data_label)
R.metric("val_examples", "Held-out", len(val_rows), "int", "hold")
R.metric("target_chars", "Characters per target", sum(target_lens) / len(rows), "num", "sky", help="with reasoning steps" if with_steps else "answer only")
R.metric("families", "Task families", len(fam), "int", "raw")
R.chart("families", "Demonstrations by family", [{"family": k, "count": v} for k, v in sorted(fam.items())], "family", [{"key": "count", "label": "Examples", "color": "kept"}], "bar")
R.chart("difficulty", "By difficulty", [{"difficulty": str(k), "count": v} for k, v in sorted(diff.items())], "difficulty", [{"key": "count", "label": "Examples", "color": "hold"}], "bar")
R.chart("target_len", "Target length (characters)", hist(target_lens, bins=20), "bin", [{"key": "count", "label": "Examples", "color": "raw"}], "bar", note="Long targets cost context and training time; short ones give the model less to learn from.")
R.chart("prompt_len", "Question length (characters)", hist(prompt_lens, bins=20), "bin", [{"key": "count", "label": "Examples", "color": "sky"}], "bar")
R.table("samples", "What the model will be trained on", [{"key": "family", "label": "Family"}, {"key": "prompt", "label": "Question (masked out of the loss)"}, {"key": "target", "label": "Target (scored)"}], [{"family": r["family"], "prompt": r["prompt"][:250], "target": r["target"][:350]} for r in rows[:20]])
R.artifact(run_dir / "train.jsonl", "train.jsonl").artifact(run_dir / "val.jsonl", "val.jsonl")
R.output("sft_train", str(run_dir / "train.jsonl")).output("sft_val", str(run_dir / "val.jsonl")).output("sft_meta", str(run_dir / "sft_meta.json"))
for k, v in model_outputs(base, "base_model").items():
    R.output(k, v)
R.output("base_accuracy", base["outputs"].get("base_accuracy"))
if prepared:
    R.note(f"Demonstrations from {data_label}; the held-out questions are its validation split, or a slice set aside from it, so evaluation never grades a training row. Base model: {base['label']} ({base['format']}).")
else:
    R.note(f"The held-out questions come from a seed far from the training seed: the same generator, never the same problems. Base model: {base['label']} ({base['format']}).")
R.save()
