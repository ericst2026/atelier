"""Your solver. Return the final numeric answer as a string."""
from atelier_nlp import hf

PROMPT = "Solve the problem step by step, then give the final answer on its own line as '#### <number>'.\n\nProblem: {q}"


def solve(question: str, generate) -> str:
    completion = generate(PROMPT.format(q=question), n=1, temperature=0.0, max_new_tokens=384)[0]
    return hf.extract_answer(completion) or ""
