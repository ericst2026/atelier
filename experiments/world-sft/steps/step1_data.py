"""Step 1 — generate demonstrations from the world."""
import json
import os
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_world import World

parse_args()
P = params({"examples": 20000, "with_steps": True, "families": ["arith", "count", "sort", "lookup", "path", "shop", "compare", "seq"], "difficulty": "mixed", "val_examples": 500, "seed": 21})
I = inputs()
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
ref = I.get("base_run_run")
if not ref:
    raise SystemExit("Choose a Pretraining run (step 3) — this experiment fine-tunes a base model.")
base = ref["outputs"]
lang = base.get("lang", "en")
families = list(P["families"]) or None
difficulty = None if P["difficulty"] == "mixed" else int(P["difficulty"])
world = World(lang=lang, seed=int(P["seed"]), families=families)

n = int(P["examples"])
rows = []
for i, ex in enumerate(world.instructions(n, with_steps=bool(P["with_steps"]), difficulty=difficulty)):
    rows.append(ex)
    if i % 2000 == 0:
        progress(70 * i / n, f"{i:,} demonstrations")
val = world.eval_set(int(P["val_examples"]), seed=555_003, difficulty=difficulty, families=families)
val_rows = [{"prompt": v["prompt"], "target": ("\n".join(v["steps"]) + "\n" if P["with_steps"] else "") + f"{world.answer_prefix} {v['answer']}", "answer": v["answer"], "family": v["family"]} for v in val]
write_jsonl(run_dir / "train.jsonl", rows)
write_jsonl(run_dir / "val.jsonl", val_rows)
(run_dir / "sft_meta.json").write_text(json.dumps({"lang": lang, "with_steps": bool(P["with_steps"]), "families": families, "system": world.system_prompt}, ensure_ascii=False, indent=2))
progress(85, "writing result")

fam = Counter(r["family"] for r in rows)
diff = Counter(r["difficulty"] for r in rows)
target_lens = [len(r["target"]) for r in rows]
prompt_lens = [len(r["prompt"]) for r in rows]

R = Result()
R.metric("examples", "Demonstrations", len(rows), "int", "kept")
R.metric("val_examples", "Held-out", len(val_rows), "int", "hold")
R.metric("target_chars", "Characters per target", sum(target_lens) / len(rows), "num", "sky", help="with reasoning steps" if P["with_steps"] else "answer only")
R.metric("families", "Task families", len(fam), "int", "raw")
R.chart("families", "Demonstrations by family", [{"family": k, "count": v} for k, v in sorted(fam.items())], "family", [{"key": "count", "label": "Examples", "color": "kept"}], "bar")
R.chart("difficulty", "By difficulty", [{"difficulty": str(k), "count": v} for k, v in sorted(diff.items())], "difficulty", [{"key": "count", "label": "Examples", "color": "hold"}], "bar")
R.chart("target_len", "Target length (characters)", hist(target_lens, bins=20), "bin", [{"key": "count", "label": "Examples", "color": "raw"}], "bar", note="Long targets cost context and training time; short ones give the model less to learn from.")
R.chart("prompt_len", "Question length (characters)", hist(prompt_lens, bins=20), "bin", [{"key": "count", "label": "Examples", "color": "sky"}], "bar")
R.table("samples", "What the model will be trained on", [{"key": "family", "label": "Family"}, {"key": "prompt", "label": "Question (masked out of the loss)"}, {"key": "target", "label": "Target (scored)"}], [{"family": r["family"], "prompt": r["prompt"][:250], "target": r["target"][:350]} for r in rows[:20]])
R.artifact(run_dir / "train.jsonl", "train.jsonl").artifact(run_dir / "val.jsonl", "val.jsonl")
R.output("sft_train", str(run_dir / "train.jsonl")).output("sft_val", str(run_dir / "val.jsonl")).output("sft_meta", str(run_dir / "sft_meta.json"))
R.output("base_model", base["model"]).output("tokenizer", base["tokenizer"]).output("lang", lang).output("base_accuracy", base.get("base_accuracy"))
R.note("The held-out questions come from a seed far from the training seed: the same generator, never the same problems.")
R.save()
