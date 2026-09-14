# Fit a curve that survives contact with a model it has not seen

`project/sweep.py` trains your grid and writes `outputs/points.json`:
`[{"params", "non_embedding", "tokens", "val_loss"}, ...]`.
`project/fit.py` exposes `fit(points) -> dict` and `predict(params, tokens) -> float`.

The grader holds back models trained at sizes your grid did not include and scores
you on how close your prediction lands. Fitting the points you already have is
easy; extrapolating is the exercise.

Choices that decide whether it works: the range of sizes (a factor of ten between
smallest and largest, at least), holding the token budget proportional to size,
scaling the learning rate with width, and deciding honestly whether the smallest
point belongs in the fit.
