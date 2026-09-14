# Measure the inside, not the output

`project/interp.py` defines `train_probe(hidden, labels)` and
`predict_probe(probe, hidden)` — `hidden` is a float array of shape
(examples, width) — and `induction_heads(model, tok)`, returning your five strongest
(layer, head) pairs, best first.

The probe is scored on held-out accuracy against a reference logistic regression.
The heads are scored against a reference ranking.

Neither is much code. The marks are for measuring the right thing: a probe that
overfits a few hundred examples, and an induction score computed one position off,
both look entirely plausible and are both wrong.
