"""Your evaluation harness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluation"))
from lib_score import option_logprobs, pick  # noqa: E402

from atelier_nlp import hf  # noqa: E402


def evaluate(model, tok, items: list[dict], task: str) -> list[dict]:
    out = []
    if task == "choice":
        for x in items:
            scores = option_logprobs(model, tok, x["question"], x["choices"], batch_size=8)
            chosen = pick(scores, "mean")
            out.append({"correct": chosen == x["answer"], "chose": chosen})
    else:
        prompts = [hf.chat_prompt(tok, x["question"] + "\nSolve step by step, then give the final answer after '####'.") for x in items]
        gens = hf.generate_batch(model, tok, prompts, 256, 0.0, batch_size=8)
        for x, g in zip(items, gens):
            out.append({"correct": hf.answers_equal(hf.extract_answer(g[0]), x["answer"])})
    return out
