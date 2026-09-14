"""Your evaluation harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "world-evaluation"))
from lib_harness import option_logprobs, pick  # noqa: E402

from atelier_mini.gen import generate  # noqa: E402
from atelier_world import World  # noqa: E402

_world = World("en", 1)


def evaluate(model, tok, items: list[dict], task: str) -> list[dict]:
    out = []
    if task == "choice":
        for x in items:
            scores = option_logprobs(model, tok, x["question"], x["options"], batch_size=16)
            chosen = pick(scores, "mean")
            out.append({"correct": chosen == x["answer"], "chose": chosen})
    else:
        gens = generate(model, tok, [x["question"] for x in items], 160, 0.0, batch_size=32)
        for x, g in zip(items, gens):
            out.append({"correct": _world.grade(g[0], x["answer"])})
    return out
