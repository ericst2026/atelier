"""Haystacks and needles, generated so they can never be in the training data."""
import random
from typing import Any


def make_haystack(world, tok, target_tokens: int, rng: random.Random) -> tuple[str, list[str]]:
    """Filler text made of the world's own stories, long enough to fill the window."""
    parts, total = [], 0
    while total < target_tokens:
        text = world.story(rng)
        parts.append(text)
        total += len(tok.encode(text)) + 1
    return ("\n".join(parts), parts)


def plant(world, tok, target_tokens: int, depth: float, rng: random.Random) -> dict[str, Any]:
    """Return a document with one fact at `depth` (0 = start, 1 = end), and the question."""
    person = rng.choice(world.pack["people"])
    obj = rng.choice(world.pack["objects"])
    place = rng.choice(world.pack["places"])
    code = rng.randint(1000, 9999)
    if world.lang == "ja":
        fact = f"{person}の{obj}の番号は{code}である。"
        question = f"{person}の{obj}の番号は何ですか。"
    else:
        fact = f"The number on {person}'s {obj} at {place} is {code}."
        question = f"What is the number on {person}'s {obj}?"
    text, parts = make_haystack(world, tok, target_tokens, rng)
    cut = max(0, min(len(parts) - 1, int(len(parts) * depth)))
    parts = parts[:cut] + [fact] + parts[cut:]
    return {"document": "\n".join(parts), "question": question, "answer": str(code), "depth": depth, "fact": fact}


def needle_prompt(doc: str, question: str, prefix: str) -> str:
    return f"{doc}\n\n{question}\n{prefix}"
