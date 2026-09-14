"""The shaped reward the guided steps use. Students replace this in their project."""
import re
from typing import Any


def repetition(text: str) -> float:
    """0 when every word is distinct, approaching 1 when the model repeats itself."""
    words = text.split()
    if len(words) < 12:
        return 0.0
    return max(0.0, 1.0 - len(set(words)) / len(words) / 0.6)


def shaped_reward(world, task: dict[str, Any], completion: str, w: dict[str, float]) -> dict[str, float]:
    """Returns the parts as well as the total, so the charts can show where reward came from."""
    correct = 1.0 if world.grade(completion, task["answer"]) else 0.0
    marked = 1.0 if world.answer_prefix in completion else 0.0
    working = 1.0 if len([l for l in completion.strip().splitlines() if l.strip()]) > 1 else 0.0
    length_pen = len(completion) / 100.0
    repeat_pen = repetition(completion)
    total = w["correct"] * correct + w["marked"] * marked + w["steps"] * working - w["length"] * length_pen - w["repeat"] * repeat_pen
    return {"total": total, "correct": correct, "marked": marked, "working": working, "length_pen": length_pen, "repeat_pen": repeat_pen}
