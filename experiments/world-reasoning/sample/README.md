# Get more right, with fewer tokens

`project/solve.py` — `solve(task, generate) -> str`. You get the question in
`task["prompt"]`; `generate(prompt, n=1, temperature=0.0, max_new_tokens=192)`
returns completions from your model and counts every token it produces.

Score: 70 × accuracy on unseen questions, plus up to 30 for efficiency — full
marks at 400 generated tokens per question, nothing at 3,000.

That trade-off is the whole exercise. Sampling eight attempts and voting will
raise accuracy and will probably cost you more than it gains. Things that tend to
pay: a short prompt that puts the model straight into working; stopping as soon as
two attempts agree; sampling again only when the first attempt produced no answer
line; a cheap sanity check on the arithmetic before committing.

`python project/run.py --model <model.pt> --tokenizer <tokenizer.json> --n 100`
runs your solver locally and prints accuracy and tokens per question.
