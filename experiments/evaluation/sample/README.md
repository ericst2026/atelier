# Write the program that produces the number

`project/harness.py` defines `evaluate(model, tok, items, task) -> list[dict]`,
one `{"correct": bool}` per item. Multiple-choice items carry `question`,
`choices` and `answer`; generation items carry `question` and `answer`.

The grader runs your harness on held-out items and compares it against a reference
implementation, item by item. Agreement is most of the marks — this is a test of
whether you built the thing correctly, not whether you built something that scores
high.

The rest is for the traps: an option that is correct but long, a question where the
prompt format changes the answer, and a generation item where the model produces
the right number in the middle of a wrong sentence.
