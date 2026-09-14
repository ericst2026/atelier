# Write a filter that a model can feel

`project/filters.py` defines `keep(doc) -> bool`, and optionally `score(doc) -> float`
if you want to rank rather than threshold. `doc` has `text` plus whatever the source
carried.

The grader scores two things: agreement with a held-out quality judgement, and the
validation loss of a fixed model trained on 5M tokens of what you kept. Keeping
everything scores badly on the first; keeping almost nothing scores badly on the
second, because there is not enough left to train on.

Start with the cheap signals — length, symbol and digit ratios, repeated lines,
whether common words appear at all — then look at what you dropped and fix the rule
that took the good documents with it. The last few points usually come from
deduplication rather than from another threshold.
