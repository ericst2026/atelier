# Solve more with fewer tokens

`project/solve.py` — `solve(question, generate) -> str`. `generate(prompt, n=1,
temperature=0.0, max_new_tokens=384)` returns `n` completions from
Qwen2.5-0.5B-Instruct and is metered by the grader.

Score: 70 × accuracy on hidden GSM8K problems + up to 30 for efficiency
(full marks at ≤ 400 generated tokens per problem, none at ≥ 3,000).

The starter asks for a chain of thought once. Try few-shot examples, majority
voting, a verification pass, splitting the problem into sub-questions, or
stopping early when two samples agree.

`python project/run.py --n 50` runs your solver on 50 training problems locally.
