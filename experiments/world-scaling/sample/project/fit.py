"""Your scaling law. L(N) = E + A·N^(−α) is a starting point, not the only option."""
import math

_state = {"E": 0.0, "A": 1.0, "alpha": 0.1}


def fit(points: list[dict]) -> dict:
    """points: [{"params", "non_embedding", "tokens", "val_loss"}]"""
    best = None
    lo = min(p["val_loss"] for p in points)
    for i in range(200):
        E = lo * i / 200.0
        xs = [math.log(p["non_embedding"]) for p in points]
        ys = []
        ok = True
        for p in points:
            r = p["val_loss"] - E
            if r <= 1e-6:
                ok = False
                break
            ys.append(math.log(r))
        if not ok:
            continue
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        intercept = my - slope * mx
        resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        cand = {"E": E, "A": math.exp(intercept), "alpha": -slope, "resid": resid}
        if best is None or cand["resid"] < best["resid"]:
            best = cand
    _state.update(best)
    return best


def predict(params: float, tokens: float) -> float:
    """Called with the non-embedding parameter count and the token budget."""
    return _state["E"] + _state["A"] * params ** (-_state["alpha"])
