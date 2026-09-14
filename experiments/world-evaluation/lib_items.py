"""Rendering benchmark items. No torch here, so step 1 runs anywhere."""
from typing import Any


def few_shot_prefix(items: list[dict[str, Any]], n: int, seed: int = 7) -> str:
    if n <= 0:
        return ""
    import random

    rng = random.Random(seed)
    shots = rng.sample(items, min(n, len(items)))
    blocks = []
    for s in shots:
        if s.get("options"):
            letters = "ABCDEF"
            body = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(s["options"]))
            blocks.append(f"{s['question']}\n{body}\nAnswer: {letters[s['answer']]}")
        else:
            blocks.append(f"{s['question']}\nAnswer: {s['answer']}")
    return "\n\n".join(blocks) + "\n\n"


def render_item(item: dict[str, Any]) -> str:
    """The benchmark item as a single string — used for the contamination experiment,
    where exactly this text is mixed into the pretraining corpus."""
    if item.get("options"):
        letters = "ABCDEF"
        body = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(item["options"]))
        return f"{item['question']}\n{body}\nAnswer: {letters[item['answer']]}"
    return f"{item['question']}\nAnswer: {item['answer']}"
