"""Loading and normalising benchmarks into two shapes: choice and generation."""
import random
from typing import Any

from atelier_nlp import hf

CHOICE, GENERATION = "choice", "generation"


def _letter_index(key: str, choices: list[str]) -> int:
    key = (key or "").strip()
    if key.isdigit():
        return int(key) - 1 if int(key) > 0 else 0
    return "ABCDE".find(key.upper()) if key[:1].upper() in "ABCDE" else 0


def load(name: str, limit: int = 500, seed: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if name == "hellaswag":
        rows = hf.dataset_split("eval", "hellaswag", "validation", limit=limit * 2)
        items = [{"kind": CHOICE, "question": r["ctx"], "choices": list(r["endings"]), "answer": int(r["label"]), "subject": "commonsense"} for r in rows if r.get("endings") and str(r.get("label", "")).isdigit()]
    elif name in ("arc-easy", "arc-challenge"):
        rows = hf.dataset_split("eval", name, "test", limit=limit * 2)
        items = []
        for r in rows:
            ch = r.get("choices") or {}
            texts = ch.get("text") if isinstance(ch, dict) else None
            labels = ch.get("label") if isinstance(ch, dict) else None
            if not texts:
                continue
            key = r.get("answerKey")
            idx = labels.index(key) if labels and key in labels else _letter_index(key, texts)
            items.append({"kind": CHOICE, "question": r["question"], "choices": list(texts), "answer": idx, "subject": name})
    elif name == "mmlu":
        rows = hf.dataset_split("eval", "mmlu", "test", limit=limit * 4)
        items = [{"kind": CHOICE, "question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]), "subject": r.get("subject", "")} for r in rows if r.get("choices")]
    elif name == "gsm8k":
        rows = hf.dataset_split("rl", "gsm8k", "test", limit=limit * 2)
        items = [{"kind": GENERATION, "question": r["question"], "answer": hf.gsm8k_gold(r["answer"]), "solution": r["answer"], "subject": "math"} for r in rows]
    else:
        raise ValueError(f"unknown benchmark {name!r}")
    rng.shuffle(items)
    return items[:limit]


def few_shot_prefix(items: list[dict], n: int, seed: int = 7) -> str:
    if n <= 0:
        return ""
    rng = random.Random(seed)
    shots = rng.sample(items, min(n, len(items)))
    out = []
    for s in shots:
        if s["kind"] == CHOICE:
            letters = "ABCD"
            body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(s["choices"][:4]))
            out.append(f"{s['question']}\n{body}\nAnswer: {letters[min(s['answer'], 3)]}")
        else:
            out.append(f"{s['question']}\n{s['solution']}")
    return "\n\n".join(out) + "\n\n"
