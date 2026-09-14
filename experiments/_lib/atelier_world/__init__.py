"""A generated world: the only data this track uses.

Everything students train on is produced here — no downloads, no licences, no
cache. Generation is deterministic in a seed, so 500M tokens regenerate
identically on any machine, and every task carries its answer *and* the reasoning
steps that reach it. That is what makes exact evaluation and a checkable reward
possible without a judge model.

    from atelier_world import World, LANGS
    w = World(lang="en", seed=7)
    for doc in w.documents(1000):  ...        # pretraining text
    for ex in w.instructions(1000): ...       # {prompt, answer, steps}
    task = w.task("arith", difficulty=2)      # one solved problem

    from atelier_world import Library
    lib = Library(w, n_docs=5000)             # a searchable corpus with known answers
"""
from .noise import KINDS as NOISE_KINDS, make_corpus, score_filter, spoil
from .retrieval import Library
from .world import LANGS, TASK_FAMILIES, World, check_answer, multiple_choice, normalize_answer

__all__ = ["World", "Library", "LANGS", "TASK_FAMILIES", "check_answer", "normalize_answer",
           "multiple_choice", "make_corpus", "score_filter", "spoil", "NOISE_KINDS"]
