"""Task families. Each returns {family, question, answer, steps, difficulty}.

`steps` is the reasoning the generator used, in the pack's language — this is what
makes rejection-sampled fine-tuning and chain-of-thought supervision possible
without any human annotation or a teacher model."""
import random
from typing import Any

FAMILIES = ["arith", "count", "sort", "lookup", "path", "shop", "compare", "seq"]
GRID = [["north", "south"], ["east", "west"]]


def _join(pack: dict, items: list[str]) -> str:
    return ("、" if not pack["spaces"] else ", ").join(items)


def _num(n: Any) -> str:
    return str(n)


def arith(rng: random.Random, pack: dict, d: int) -> dict:
    terms = 2 + min(d, 3)
    hi = [10, 30, 99, 999][min(d, 3)]
    nums = [rng.randint(2, hi) for _ in range(terms)]
    ops = [rng.choice(["+", "-"]) if d < 2 else rng.choice(["+", "-", "*"]) for _ in range(terms - 1)]
    if any(o == "*" for o in ops):
        nums = [n if n < 30 else n % 29 + 2 for n in nums]
    expr, value, steps = _num(nums[0]), nums[0], []
    for i, (op, n) in enumerate(zip(ops, nums[1:])):
        prev = value
        value = value + n if op == "+" else value - n if op == "-" else value * n
        piece = f"{prev} {op} {n}"
        steps.append((pack["s_first"] if not steps else pack["s_then"]).format(a=piece, b=value))
        # parenthesised so the printed expression means what the steps compute;
        # mixing + and * without brackets would contradict normal precedence
        expr = f"({expr} {op} {n})" if i < len(ops) - 1 and "*" in ops else f"{expr} {op} {n}"
    steps.append(pack["s_answer"].format(b=value))
    return {"family": "arith", "question": rng.choice(pack["q_arith"]).format(expr=expr), "answer": str(value), "steps": steps, "difficulty": d}


def count(rng: random.Random, pack: dict, d: int) -> dict:
    good = rng.choice(pack["goods"])
    groups = 2 + min(d, 3)
    counts = [rng.randint(1, 6 + 4 * d) for _ in range(groups)]
    places = rng.sample(pack["places"], groups)
    lines = [f"{p}: {c}" for p, c in zip(places, counts)]
    total = sum(counts)
    steps = [pack["s_count"].format(g=good, items=_join(pack, [str(c) for c in counts]), b=total), pack["s_answer"].format(b=total)]
    q = _join(pack, lines) + ("\n" if pack["spaces"] else "\n") + rng.choice(pack["q_count"]).format(g=good)
    return {"family": "count", "question": q, "answer": str(total), "steps": steps, "difficulty": d}


def sort_task(rng: random.Random, pack: dict, d: int) -> dict:
    n = 3 + min(d, 4)
    nums = rng.sample(range(1, 20 + 30 * d), n)
    ordered = sorted(nums)
    steps = [pack["s_first"].format(a=pack["yes"], b=str(min(nums)))] if False else []
    remaining = list(nums)
    picked = []
    for _ in range(n):
        m = min(remaining)
        remaining.remove(m)
        picked.append(m)
        steps.append(pack["s_then"].format(a=_join(pack, [str(x) for x in remaining]) or "-", b=str(m)))
    ans = _join(pack, [str(x) for x in ordered])
    steps.append(pack["s_answer"].format(b=ans))
    return {"family": "sort", "question": rng.choice(pack["q_sort"]).format(items=_join(pack, [str(x) for x in nums])), "answer": ans, "steps": steps[-min(len(steps), 4):], "difficulty": d}


def lookup(rng: random.Random, pack: dict, d: int) -> dict:
    n = 3 + min(d, 4)
    people = rng.sample(pack["people"], n)
    places = rng.sample(pack["places"], n)
    objects = rng.sample(pack["objects"], n)
    facts = [pack["fact"][1].format(p=p, o=o, l=l, q="", n="", u="") for p, o, l in zip(people, objects, places)]
    i = rng.randrange(n)
    q = pack["q_lookup"][0].format(facts="\n".join(facts), p=people[i], o=objects[i], l=places[i])
    steps = [facts[i], pack["s_answer"].format(b=places[i])]
    return {"family": "lookup", "question": q, "answer": places[i], "steps": steps, "difficulty": d}


def path(rng: random.Random, pack: dict, d: int) -> dict:
    """A 3×3 grid of places; moves walk between them."""
    grid = rng.sample(pack["places"], 9)
    r, c = rng.randrange(3), rng.randrange(3)
    start = grid[r * 3 + c]
    moves, steps = [], []
    for _ in range(2 + min(d, 3)):
        options = []
        if r > 0:
            options.append("north")
        if r < 2:
            options.append("south")
        if c > 0:
            options.append("west")
        if c < 2:
            options.append("east")
        mv = rng.choice(options)
        before = grid[r * 3 + c]
        r, c = (r - 1, c) if mv == "north" else (r + 1, c) if mv == "south" else (r, c - 1) if mv == "west" else (r, c + 1)
        after = grid[r * 3 + c]
        moves.append(pack["directions"][mv])
        steps.append(pack["s_move"].format(a=before, d=pack["directions"][mv], b=after))
    end = grid[r * 3 + c]
    steps.append(pack["s_answer"].format(b=end))
    layout = _join(pack, [f"{grid[i * 3 + j]}" for i in range(3) for j in range(3)])
    q = f"{layout}\n" + pack["q_path"][0].format(p=rng.choice(pack["people"]), l0=start, moves=_join(pack, moves))
    return {"family": "path", "question": q, "answer": end, "steps": steps, "difficulty": d}


def shop(rng: random.Random, pack: dict, d: int) -> dict:
    g1, g2 = rng.sample(pack["goods"], 2)
    n1, n2 = rng.randint(2, 4 + 3 * d), rng.randint(2, 4 + 3 * d)
    c1, c2 = rng.randint(2, 9 + 5 * d), rng.randint(2, 9 + 5 * d)
    a, b = n1 * c1, n2 * c2
    total = a + b
    steps = [
        pack["s_cost"].format(n=n1, c=f"{g1} ({c1})", b=a),
        pack["s_cost"].format(n=n2, c=f"{g2} ({c2})", b=b),
        pack["s_then"].format(a=f"{a} + {b}", b=total),
        pack["s_answer"].format(b=total),
    ]
    q = pack["q_shop"][0].format(p=rng.choice(pack["people"]), n1=n1, g1=g1, c1=c1, n2=n2, g2=g2, c2=c2)
    return {"family": "shop", "question": q, "answer": str(total), "steps": steps, "difficulty": d}


def compare(rng: random.Random, pack: dict, d: int) -> dict:
    p, q_ = rng.sample(pack["people"], 2)
    u = rng.choice(pack["units"])
    n1, n2 = rng.randint(1, 20 + 30 * d), rng.randint(1, 20 + 30 * d)
    if n1 == n2:
        n2 += 1
    more = p if n1 > n2 else q_
    diff = abs(n1 - n2)
    steps = [pack["s_then"].format(a=f"{max(n1, n2)} - {min(n1, n2)}", b=diff), pack["s_answer"].format(b=diff)]
    return {"family": "compare", "question": pack["q_compare"][0].format(p=p, q=q_, u=u, n1=n1, n2=n2, more=more), "answer": str(diff), "steps": steps, "difficulty": d}


def seq(rng: random.Random, pack: dict, d: int) -> dict:
    kind = rng.choice(["add", "add", "mul"] if d >= 2 else ["add"])
    start = rng.randint(1, 9 + 5 * d)
    step = rng.randint(2, 5 + 4 * d)
    n = 4 + min(d, 2)
    if kind == "add":
        items = [start + i * step for i in range(n)]
        rule = f"+{step}"
    else:
        step = rng.randint(2, 3)
        items = [start * step ** i for i in range(n)]
        rule = f"×{step}"
    nxt = items[-1] + step if kind == "add" else items[-1] * step
    steps = [pack["s_first"].format(a=f"{items[1]} - {items[0]}" if kind == "add" else f"{items[1]} / {items[0]}", b=step), pack["s_then"].format(a=f"{items[-1]} {rule}", b=nxt), pack["s_answer"].format(b=nxt)]
    return {"family": "seq", "question": rng.choice(pack["q_seq"]).format(items=_join(pack, [str(x) for x in items])), "answer": str(nxt), "steps": steps, "difficulty": d}


GENERATORS = {"arith": arith, "count": count, "sort": sort_task, "lookup": lookup, "path": path, "shop": shop, "compare": compare, "seq": seq}


def generate(family: str, rng: random.Random, pack: dict, difficulty: int) -> dict:
    return GENERATORS[family](rng, pack, max(0, min(int(difficulty), 3)))
