"""Your filter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "curation"))
from lib_quality import signals  # noqa: E402


def keep(doc: dict) -> bool:
    text = doc.get("text") or ""
    s = signals(text)
    return (
        s["chars"] >= 300
        and 3.0 <= s["mean_word"] <= 10.0
        and s["symbol_ratio"] < 0.15
        and s["digit_ratio"] < 0.2
        and s["dup_line_ratio"] < 0.3
        and s["stopword_hits"] >= 2
    )


def score(doc: dict) -> float:
    """Optional: a continuous quality estimate, used to rank when a budget binds."""
    s = signals(doc.get("text") or "")
    return s["unique_word_ratio"] - s["symbol_ratio"] - s["dup_line_ratio"]
