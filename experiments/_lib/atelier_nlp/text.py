import re
import unicodedata
from typing import Iterable

# GPT-2 style pre-tokenizer: optional leading space + letters | digits | other, whitespace kept separately.
PRETOK = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def pretokenize(text: str) -> list[str]:
    return PRETOK.findall(text)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def char_class(ch: str) -> str:
    if ch.isspace():
        return "spaces"
    if ch.isdigit():
        return "digits"
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        o = ord(ch)
        if 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF or 0x3040 <= o <= 0x30FF:
            return "cjk"
        return "letters"
    if cat.startswith("P") or cat.startswith("S"):
        return "punctuation"
    return "other"


def char_classes(texts: Iterable[str]) -> dict[str, int]:
    counts = {"letters": 0, "digits": 0, "spaces": 0, "punctuation": 0, "cjk": 0, "other": 0}
    for t in texts:
        for ch in t:
            counts[char_class(ch)] += 1
    return counts


def normalize(text: str, lowercase: bool = True, collapse_ws: bool = True, strip_punct: bool = False) -> str:
    t = unicodedata.normalize("NFKC", text)
    if lowercase:
        t = t.lower()
    if strip_punct:
        t = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in t)
    if collapse_ws:
        t = " ".join(t.split())
    return t
