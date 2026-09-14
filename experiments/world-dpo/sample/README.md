# Build the pairs, then live with what they teach

`project/pairs.py` — `build_pairs(world, model, tok, generate) -> list[dict]` of
`{"prompt", "chosen", "rejected"}`. `project/train.py` trains with them.

The labels are the experiment. Correctness is the obvious basis and works. Less
obvious and often better: pair a correct answer against a *nearly* correct one
rather than a wildly wrong one, so the gradient carries information about the
specific mistake. Or pair on agreement between samples, on brevity at equal
correctness, or on whether the working actually leads to the stated answer.

Watch what you are teaching implicitly. If your chosen answers are always shorter,
the model learns brevity as much as correctness — and the grader checks that it
can still write a complete answer.
