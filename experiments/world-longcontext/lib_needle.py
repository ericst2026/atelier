"""Haystacks and needles, generated so they can never be in the training data.

The filler is the world's own stories by default. With `passages` (paragraphs from a
prepared documents dataset, see `split_passages`) the filler is drawn from those
instead; the planted fact is still generated, so it is never in any training text."""
import random
from typing import Any, Optional


def split_passages(texts: list[str], max_chars: int = 600) -> list[str]:
    """Documents cut into paragraphs of at most `max_chars`, so a needle can be planted
    at a fine depth in a haystack built from them."""
    out: list[str] = []
    for text in texts:
        for para in text.splitlines():
            para = " ".join(para.split())
            while len(para) > max_chars:
                cut = para.rfind(". ", max_chars // 3, max_chars)
                cut = cut + 1 if cut != -1 else para.rfind(" ", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                out.append(para[:cut].strip())
                para = para[cut:].strip()
            if para:
                out.append(para)
    return out


def make_haystack(world, tok, target_tokens: int, rng: random.Random, passages: Optional[list[str]] = None) -> tuple[str, list[str]]:
    """Filler text long enough to fill the window: the world's stories, or prepared passages."""
    parts, total = [], 0
    start = rng.randrange(len(passages)) if passages else 0
    while total < target_tokens:
        # prepared passages are taken in their own order from a random start, so the
        # haystack reads as a stretch of real text rather than shuffled sentences
        text = passages[(start + len(parts)) % len(passages)] if passages else world.story(rng)
        parts.append(text)
        total += len(tok.encode(text)) + 1
    return ("\n".join(parts), parts)


def plant(world, tok, target_tokens: int, depth: float, rng: random.Random, passages: Optional[list[str]] = None) -> dict[str, Any]:
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
    text, parts = make_haystack(world, tok, target_tokens, rng, passages)
    cut = max(0, min(len(parts) - 1, int(len(parts) * depth)))
    parts = parts[:cut] + [fact] + parts[cut:]
    return {"document": "\n".join(parts), "question": question, "answer": str(code), "depth": depth, "fact": fact}


def needle_prompt(doc: str, question: str, prefix: str) -> str:
    return f"{doc}\n\n{question}\n{prefix}"
