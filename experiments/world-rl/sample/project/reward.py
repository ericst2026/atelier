"""Your reward. Higher is better; GRPO normalises within each group."""
from atelier_world import World

_world = {}


def _w(lang: str = "en") -> World:
    if lang not in _world:
        _world[lang] = World(lang=lang, seed=0)
    return _world[lang]


def reward(task: dict, completion: str) -> float:
    world = _w(task.get("lang", "en"))
    r = 1.0 if world.grade(completion, task["answer"]) else 0.0
    if world.answer_prefix in completion:
        r += 0.2
    r -= 0.05 * len(completion) / 100.0
    return r
