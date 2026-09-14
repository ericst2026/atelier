"""Cheap per-document signals, and the rule set the guided step exposes as sliders."""
import re
from collections import Counter

STOPWORDS = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "with", "as", "was", "on", "are", "this", "at", "by"}
JA_COMMON = {"は", "を", "に", "の", "が", "で", "と", "た", "て", "だ"}
WORD = re.compile(r"[A-Za-z']+")


def signals(text: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    words = WORD.findall(text)
    n_chars = max(len(text), 1)
    n_words = max(len(words), 1)
    counts = Counter(l.strip() for l in lines)
    return {
        "chars": len(text),
        "words": len(words),
        "lines": len(lines),
        "mean_word": sum(len(w) for w in words) / n_words,
        "symbol_ratio": sum(1 for c in text if not (c.isalnum() or c.isspace())) / n_chars,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
        "upper_ratio": sum(1 for c in text if c.isupper()) / max(sum(1 for c in text if c.isalpha()), 1),
        "dup_line_ratio": sum(c for c in counts.values() if c > 1) / max(len(lines), 1),
        "ellipsis_ratio": sum(1 for l in lines if l.rstrip().endswith(("…", "..."))) / max(len(lines), 1),
        "stopword_hits": len({w.lower() for w in words} & STOPWORDS) + len(set(text) & JA_COMMON),
        "unique_word_ratio": len({w.lower() for w in words}) / n_words,
    }


RULES = ("too short", "symbols", "digits", "shouting", "repeated lines", "not prose", "truncated", "few distinct words")


def apply_rules(text: str, p: dict) -> tuple[bool, str]:
    """(keep, the first rule that rejected it) — so the UI can show what each costs."""
    s = signals(text)
    checks = [
        ("too short", s["chars"] < p["min_chars"]),
        ("symbols", s["symbol_ratio"] > p["max_symbol_ratio"]),
        ("digits", s["digit_ratio"] > p["max_digit_ratio"]),
        ("shouting", s["upper_ratio"] > p["max_upper_ratio"]),
        ("repeated lines", s["dup_line_ratio"] > p["max_dup_line_ratio"]),
        ("not prose", s["stopword_hits"] < p["min_stopword_hits"]),
        ("truncated", s["ellipsis_ratio"] > p["max_ellipsis_ratio"]),
        ("few distinct words", s["unique_word_ratio"] < p["min_unique_word_ratio"]),
    ]
    for name, failed in checks:
        if failed:
            return False, name
    return True, ""
