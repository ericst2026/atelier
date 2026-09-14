# A filter judged twice

`project/filters.py` defines `keep(doc) -> bool`. `doc` has `text`, and nothing else
— the labels stay with the grader.

You are scored on three things: F1 against those labels, the validation loss of a
fixed model trained on what you kept, and whether you kept enough text to train on
at all. Keeping everything fails the first. Keeping almost nothing fails the third,
and usually the second too.

The nine kinds of damage are in `experiments/_lib/atelier_world/noise.py` — read it
once, then write rules without looking at it again. Most are separable on one axis;
two are not. Duplicates cannot be found by looking at a document at all, so if you
want them, your `keep` needs to remember what it has already seen.
