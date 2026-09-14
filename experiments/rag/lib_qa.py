"""Answer scoring for extractive question answering, and the prompt the reader sees."""
import re
import string
from collections import Counter

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = str.maketrans("", "", string.punctuation)
REFUSALS = ("no answer", "not in", "cannot", "can't", "unanswerable", "does not", "doesn't", "unknown", "not mentioned", "not provided", "insufficient")


def normalize(s: str) -> str:
    s = (s or "").lower().translate(PUNCT)
    return " ".join(ARTICLES.sub(" ", s).split())


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    return any(p == normalize(g) for g in golds if g)


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p = normalize(pred).split()
    for g in golds:
        gt = normalize(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = Counter(p) & Counter(gt)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision, recall = overlap / len(p), overlap / len(gt)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def is_refusal(pred: str) -> bool:
    p = (pred or "").lower().strip()
    return not p or any(r in p[:120] for r in REFUSALS)


SYSTEM = "Answer the question using only the passages provided. Answer in as few words as possible. If the passages do not contain the answer, reply exactly: no answer."


def build_prompt(question: str, passages: list[dict]) -> str:
    if not passages:
        return f"Question: {question}\nAnswer:"
    body = "\n\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))
    return f"{body}\n\nQuestion: {question}\nAnswer:"
