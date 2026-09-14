"""Corrupting a clean corpus in known ways.

Real web text is dirty in a handful of recognisable ways, and a curation filter is
judged on whether it removes those without removing anything else. Here every
document carries a label saying how it was spoiled — so a filter can be scored on
precision and recall directly, which no real corpus allows, as well as on the
validation loss of a model trained on what survived."""
import random
from typing import Any

KINDS = ("clean", "boilerplate", "repeated", "symbols", "truncated", "shouting", "listing", "gibberish", "tiny", "duplicate")

NAV = ["Home | About | Contact | Privacy | Terms", "Skip to main content", "Cookies help us deliver our services.",
       "Share on social media", "Click here to read more", "Sign up for our newsletter", "All rights reserved.",
       "You are here: Home › Articles ›", "Related posts you may like", "Advertisement"]


def spoil(text: str, kind: str, rng: random.Random) -> str:
    """Return the document as it would look after one kind of damage."""
    if kind == "clean":
        return text
    if kind == "boilerplate":
        # a paragraph of real text drowned in navigation furniture
        head = "\n".join(rng.sample(NAV, k=min(5, len(NAV))))
        tail = "\n".join(rng.sample(NAV, k=3))
        return f"{head}\n{text}\n{tail}"
    if kind == "repeated":
        line = rng.choice([l for l in text.splitlines() if l.strip()] or [text])
        return "\n".join([line] * rng.randint(8, 25))
    if kind == "symbols":
        junk = "".join(rng.choice("#*<>{}[]|/\\~^=+_@$%") for _ in range(len(text) // 2))
        words = text.split()
        return " ".join(w + rng.choice(["", "", "###", "|||", ">>>"]) for w in words) + junk
    if kind == "truncated":
        parts = [p[: rng.randint(20, 60)] + "…" for p in text.split(".") if p.strip()]
        return "\n".join(parts[: rng.randint(5, 15)])
    if kind == "shouting":
        return text.upper()
    if kind == "listing":
        # a product grid: numbers and short fragments, almost no prose
        return "\n".join(f"SKU-{rng.randint(10000, 99999)} · {rng.randint(2, 900)} · {rng.choice(['in stock', 'out of stock', '—'])}" for _ in range(rng.randint(10, 30)))
    if kind == "gibberish":
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return " ".join("".join(rng.choice(alphabet) for _ in range(rng.randint(3, 12))) for _ in range(rng.randint(40, 120)))
    if kind == "tiny":
        return " ".join(text.split()[: rng.randint(3, 12)])
    return text


def make_corpus(world, n: int, seed: int = 17, dirty_share: float = 0.55, duplicate_share: float = 0.12, mix: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """A corpus where the label is known.

    Each document is {text, kind, clean}: `kind` is how it was spoiled, `clean` is
    whether a good filter should keep it. Duplicates are exact or near copies of a
    document already in the corpus, which is a different problem from dirt and is
    labelled separately."""
    rng = random.Random(seed)
    dirty_kinds = [k for k in KINDS if k not in ("clean", "duplicate")]
    weights = mix or {k: 1.0 for k in dirty_kinds}
    kinds = list(weights)
    ws = [weights[k] for k in kinds]

    docs: list[dict[str, Any]] = []
    originals = list(world.documents(max(1, int(n * (1 - duplicate_share)))))
    for i, d in enumerate(originals):
        if rng.random() < dirty_share:
            kind = rng.choices(kinds, ws)[0]
            docs.append({"id": len(docs), "text": spoil(d["text"], kind, rng), "kind": kind, "clean": False, "source": d.get("kind")})
        else:
            docs.append({"id": len(docs), "text": d["text"], "kind": "clean", "clean": True, "source": d.get("kind")})

    n_dupes = n - len(docs)
    pool = [d for d in docs if d["clean"]] or docs
    for _ in range(max(0, n_dupes)):
        src = rng.choice(pool)
        text = src["text"]
        if rng.random() < 0.5:
            words = text.split(" ") if world.pack["spaces"] else list(text)
            if len(words) > 4:
                j = rng.randrange(len(words) - 1)
                words[j], words[j + 1] = words[j + 1], words[j]
            text = (" " if world.pack["spaces"] else "").join(words)
        docs.append({"id": len(docs), "text": text, "kind": "duplicate", "clean": False, "dup_of": src["id"], "source": src.get("source")})
    rng.shuffle(docs)
    for i, d in enumerate(docs):
        d["id"] = i
    return docs


def score_filter(docs: list[dict[str, Any]], kept: list[bool]) -> dict[str, Any]:
    """Precision and recall against the labels, plus a breakdown by kind."""
    tp = sum(1 for d, k in zip(docs, kept) if k and d["clean"])
    fp = sum(1 for d, k in zip(docs, kept) if k and not d["clean"])
    fn = sum(1 for d, k in zip(docs, kept) if not k and d["clean"])
    by_kind: dict[str, list[int]] = {}
    for d, k in zip(docs, kept):
        e = by_kind.setdefault(d["kind"], [0, 0])
        e[0] += int(k)
        e[1] += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "kept": sum(kept),
        "by_kind": {k: v[0] / v[1] for k, v in sorted(by_kind.items())},
        "counts": {k: v[1] for k, v in sorted(by_kind.items())},
    }
