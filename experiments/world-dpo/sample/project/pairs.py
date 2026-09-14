"""Your pair construction."""


def build_pairs(world, model, tok, generate) -> list[dict]:
    """generate(prompts, n, temperature) -> list of lists of completions."""
    tasks = world.eval_set(2000, seed=707_001)
    groups = generate([t["prompt"] for t in tasks], 6, 1.0)
    pairs = []
    for t, group in zip(tasks, groups):
        right = [c for c in group if world.grade(c, t["answer"])]
        wrong = [c for c in group if not world.grade(c, t["answer"])]
        if right and wrong:
            pairs.append({"prompt": t["prompt"], "chosen": min(right, key=len).strip(), "rejected": max(wrong, key=len).strip()})
    return pairs
