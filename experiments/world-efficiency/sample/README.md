# Make it fast without making it wrong

`project/optimise.py` defines `optimise(model, tok) -> object` with a
`generate(prompts, max_new_tokens) -> list[str]` method. Quantise the weights,
add a cache, speculate with a draft, distil a smaller student, or combine them.

The grader measures accuracy on unseen questions, tokens per second, and the memory
your object holds. Speed with no accuracy scores nothing; accuracy with no speedup
scores nothing either. The combination is the point.

Roughly what each is worth on this model: the key/value cache is the largest single
win and changes nothing about the output. int8 halves the memory for about a point
of accuracy; int4 needs per-channel scales and the output layer left alone.
Speculation pays only when the draft agrees most of the time, which usually means a
draft distilled from the same model rather than a smaller one trained separately.
