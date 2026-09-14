"""Step 4 — held-out loss and side-by-side generations for base vs tuned, optionally judged."""
import os
import re
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress
from atelier_nlp import hf

parse_args()
P = params({"n_prompts": 40, "max_new_tokens": 256, "temperature": 0.0, "judge": True, "judge_model": "Qwen2.5-1.5B-Instruct"})
I = inputs()
val = hf.read_jsonl(I["val"], limit=max(int(P["n_prompts"]), 100))
base_path = Path(I["base_model"])
tok = hf.load_tokenizer(base_path)
if not getattr(tok, "chat_template", None):
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
prompts_msgs = [[m for m in r["messages"] if m["role"] != "assistant"][: len(r["messages"]) - 1] for r in val[: int(P["n_prompts"])]]
prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompts_msgs]
refs = [r["messages"][-1]["content"] for r in val[: int(P["n_prompts"])]]

results = {}
for name, adapter in (("base", None), ("tuned", I["model_dir"] if I["method"] == "lora" else None)):
    path = base_path if name == "base" or I["method"] == "lora" else Path(I["model_dir"])
    model = hf.load_model(path, adapter=adapter)
    progress(10 if name == "base" else 45, f"{name}: held-out loss")
    loss = hf.eval_loss(model, tok, val[:100], max_length=int(I.get("max_length", 1024)))
    progress(20 if name == "base" else 55, f"{name}: generating {len(prompts)} answers")
    gens = hf.generate_batch(model, tok, prompts, int(P["max_new_tokens"]), float(P["temperature"]), progress=lambda d, t: progress((20 if name == "base" else 55) + 20 * d / t, f"{name}: {d}/{t}"))
    results[name] = {"loss": loss, "answers": [g[0].strip() for g in gens]}
    del model
    import torch

    torch.cuda.empty_cache()

verdicts = []
if bool(P["judge"]):
    jt = hf.load_tokenizer(hf.model_path(P["judge_model"]))
    judge = hf.load_model(hf.model_path(P["judge_model"]))
    jp = []
    for i, m in enumerate(prompts_msgs):
        q = m[-1]["content"]
        a, b = results["base"]["answers"][i], results["tuned"]["answers"][i]
        rubric = f"You are grading two answers to the same request. Reply with exactly one word: A, B, or TIE.\n\nRequest:\n{q[:1500]}\n\nAnswer A:\n{a[:1500]}\n\nAnswer B:\n{b[:1500]}\n\nWhich answer is more helpful, correct and well-written?"
        jp.append(hf.chat_prompt(jt, rubric))
    out = hf.generate_batch(judge, jt, jp, 8, 0.0, progress=lambda d, t: progress(80 + 15 * d / t, f"judge: {d}/{t}"))
    for g in out:
        w = re.findall(r"\b(A|B|TIE)\b", g[0].upper())
        verdicts.append({"A": "base", "B": "tuned", "TIE": "tie"}[w[0]] if w else "tie")
wins = verdicts.count("tuned")
losses_ = verdicts.count("base")
ties = verdicts.count("tie")

R = Result()
R.metric("base_loss", "Base held-out loss", results["base"]["loss"], "num", "raw")
R.metric("tuned_loss", "Tuned held-out loss", results["tuned"]["loss"], "num", "kept")
if verdicts:
    R.metric("win_rate", "Judge prefers tuned", wins / len(verdicts), "pct", "kept", help=f"{wins} wins · {ties} ties · {losses_} losses")
R.metric("tuned_len", "Tuned answer length (chars)", sum(len(a) for a in results["tuned"]["answers"]) / len(prompts), "num", "sky")
R.chart("loss", "Held-out loss", [{"model": "base", "loss": results["base"]["loss"]}, {"model": "tuned", "loss": results["tuned"]["loss"]}], "model", [{"key": "loss", "label": "Loss", "color": "hold"}], "bar")
if verdicts:
    R.chart("judge", "Judge verdicts", [{"verdict": "tuned wins", "n": wins}, {"verdict": "tie", "n": ties}, {"verdict": "base wins", "n": losses_}], "verdict", [{"key": "n", "label": "Prompts", "color": "kept"}], "bar")
R.chart("lengths", "Answer lengths", [dict(b, **{"base": b["count"]}) for b in hist([len(a) for a in results["base"]["answers"]], bins=15)], "bin", [{"key": "base", "label": "Base", "color": "raw"}], "bar", note="Base model answers; fine-tuned answers are in the table.")
R.table("side", "Side by side", [{"key": "prompt", "label": "Prompt"}, {"key": "base", "label": "Base"}, {"key": "tuned", "label": "Tuned"}, {"key": "ref", "label": "Reference"}, {"key": "verdict", "label": "Judge"}], [{"prompt": m[-1]["content"][:400], "base": results["base"]["answers"][i][:600], "tuned": results["tuned"]["answers"][i][:600], "ref": refs[i][:400], "verdict": verdicts[i] if verdicts else ""} for i, m in enumerate(prompts_msgs)])
R.output("win_rate", wins / len(verdicts) if verdicts else None).output("tuned_loss", results["tuned"]["loss"])
R.save()
