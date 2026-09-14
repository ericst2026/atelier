"""Your tools. Keep them narrow: a tool the model cannot call correctly is worse than none."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "world-tools"))
from lib_tools import calc, count, make_lookup, render as _render  # noqa: E402

TOOLS = {"calc": calc, "count": count, "lookup": make_lookup({})}


def render(name: str, args: str, result: str) -> str:
    """How a completed call looks in the context the model reads."""
    return _render(name, args, result)
