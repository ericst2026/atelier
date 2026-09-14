"""Cheap text-quality signals. Every one is a heuristic somebody found useful and
somebody else found harmful; measuring is the only way to tell which you have."""
import re
from collections import Counter

STOPWORDS = {"the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "with", "as", "was", "on", "are", "this"}
WORD = re.compile(r"[A-Za-z']+")


def signals(text: str) -> dict:
    lines = [l for l in text.splitlines() if l.strip()]
    words = WORD.findall(text)
    n_chars = max(len(text), 1)
    n_words = max(len(words), 1)
    counts = Counter(l.strip() for l in lines)
    dup_lines = sum(c for c in counts.values() if c > 1)
    ends_ellipsis = sum(1 for l in lines if l.rstrip().endswith(("…", "...")))
    return {
        "chars": len(text),
        "words": len(words),
        "lines": len(lines),
        "mean_word": sum(len(w) for w in words) / n_words,
        "symbol_ratio": sum(1 for c in text if not (c.isalnum() or c.isspace())) / n_chars,
        "digit_ratio": sum(1 for c in text if c.isdigit()) / n_chars,
        "upper_ratio": sum(1 for c in text if c.isupper()) / n_chars,
        "dup_line_ratio": dup_lines / max(len(lines), 1),
        "ellipsis_ratio": ends_ellipsis / max(len(lines), 1),
        "stopword_hits": len({w.lower() for w in words} & STOPWORDS),
        "unique_word_ratio": len({w.lower() for w in words}) / n_words,
    }


def default_keep(text: str, p: dict) -> tuple[bool, str]:
    """Returns (keep, the first rule that rejected it) so the UI can show what each rule costs."""
    s = signals(text)
    checks = [
        ("too short", s["chars"] < p["min_chars"]),
        ("too long", s["chars"] > p["max_chars"]),
        ("word length", not (p["min_mean_word"] <= s["mean_word"] <= p["max_mean_word"])),
        ("symbols", s["symbol_ratio"] > p["max_symbol_ratio"]),
        ("digits", s["digit_ratio"] > p["max_digit_ratio"]),
        ("repeated lines", s["dup_line_ratio"] > p["max_dup_line_ratio"]),
        ("not prose", s["stopword_hits"] < p["min_stopword_hits"]),
        ("truncated lines", s["ellipsis_ratio"] > p["max_ellipsis_lines"]),
    ]
    for name, failed in checks:
        if failed:
            return False, name
    return True, ""
