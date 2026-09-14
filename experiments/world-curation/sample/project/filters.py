"""Your filter. Called once per document, in corpus order."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "world-curation"))
from lib_signals import signals  # noqa: E402


def keep(doc: dict) -> bool:
    s = signals(doc.get("text") or "")
    return (
        s["chars"] >= 120
        and s["symbol_ratio"] < 0.2
        and s["digit_ratio"] < 0.25
        and s["upper_ratio"] < 0.4
        and s["dup_line_ratio"] < 0.4
        and s["stopword_hits"] >= 2
        and s["unique_word_ratio"] >= 0.35
    )
