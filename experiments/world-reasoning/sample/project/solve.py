"""Your solver. Return the final answer as a string."""
from atelier_world import World

_worlds = {}


def _world(lang="en"):
    if lang not in _worlds:
        _worlds[lang] = World(lang=lang, seed=0)
    return _worlds[lang]


def solve(task: dict, generate) -> str:
    world = _world(task.get("lang", "en"))
    text = generate(task["prompt"], n=1, temperature=0.0, max_new_tokens=192)[0]
    return world.extract(text) or ""
