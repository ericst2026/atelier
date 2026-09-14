"""Your reward. Higher is better; the trainer normalises within each group."""
from atelier_nlp import hf


def reward(question: str, completion: str, gold: str) -> float:
    pred = hf.extract_answer(completion)
    r = 1.0 if hf.answers_equal(pred, gold) else 0.0
    r += hf.format_reward(completion)  # 0.2 for a clearly marked final answer
    return r
