# Quality per trained weight

`project/adapt.py` exposes `adapt(model) -> dict` — apply whatever
parameter-efficient scheme you like, return a census containing at least
`{"trainable"}` — and trains, saving `outputs/model.pt`.

The grader divides accuracy on unseen questions by the trainable parameters you
reported. LoRA at a well-chosen rank is the obvious answer. Others that work, and
sometimes better on a small model: adapting only the last two blocks, training only
the normalisation scales and biases, a low rank on attention and a higher one on
the MLP, or a trained prompt prefix with the model entirely frozen.

Report the count honestly — the grader reads your number, and a false one is easy
to spot next to the accuracy it claims to have bought.
