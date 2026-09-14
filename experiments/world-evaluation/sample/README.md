# Write the program that produces the number

`project/harness.py` defines `evaluate(model, tok, items, task) -> list[dict]`, one
`{"correct": bool}` per item. Choice items carry `question`, `options` and `answer`;
open items carry `question` and `answer`.

The grader runs your harness on held-out items and compares it with a reference,
item by item. Agreement is most of the marks: this tests whether you built the thing
correctly, not whether it scores high.

The traps are the ones that catch real harnesses. An option that is correct but long.
A prompt format that changes the answer. A generated response with the right number
in the middle of a wrong sentence. And the one people get wrong most often: scoring
the option tokens only, rather than the whole sequence including the question.
