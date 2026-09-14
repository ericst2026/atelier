# Measure the inside, not the output

`project/interp.py` defines two things.

`train_probe(hidden, labels)` and `predict_probe(probe, hidden)` — `hidden` is a
float array of shape (examples, width). Scored on held-out accuracy against a
reference logistic regression, on a property the grader chooses.

`induction_heads(model, tok)` — your five strongest (layer, head) pairs. Scored
against a reference ranking, so what matters is measuring the right thing: feed a
repeated random sequence and score each head by how much attention lands on the
token that followed the same token the first time round.

Neither is much code. The marks are for the measurement being correct — a probe
that overfits a few hundred examples, or an induction score computed against the
wrong offset by one, both look plausible and are both wrong.
