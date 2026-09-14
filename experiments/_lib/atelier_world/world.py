"""The world: documents for pretraining, instruction pairs for fine-tuning,
and solved tasks for RL and reasoning — all from a seed."""
import random
import re
from typing import Any, Iterator, Optional

from . import tasks
from .lang_en import PACK as EN
from .lang_ja import PACK as JA

LANGS = {"en": EN, "ja": JA}
TASK_FAMILIES = tasks.FAMILIES
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_answer(text: str) -> str:
    t = (text or "").strip().replace(",", "").replace("，", "").replace("。", "").rstrip(".")
    return " ".join(t.split())


def extract_answer(text: str, pack: dict) -> Optional[str]:
    """The line after the answer marker, else the last number, else the last line."""
    prefix = pack["answer_prefix"]
    if prefix in text:
        tail = text.split(prefix)[-1].strip().splitlines()
        if tail:
            return normalize_answer(tail[0])
    nums = _NUM.findall(text)
    if nums:
        return normalize_answer(nums[-1])
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return normalize_answer(lines[-1]) if lines else None


def check_answer(prediction: Optional[str], gold: str) -> bool:
    if prediction is None:
        return False
    p, g = normalize_answer(prediction), normalize_answer(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.replace(" ", "") == g.replace(" ", "")


class World:
    def __init__(self, lang: str = "en", seed: int = 7, families: Optional[list[str]] = None):
        if lang not in LANGS:
            raise ValueError(f"unknown language {lang!r}; available: {sorted(LANGS)}")
        self.pack = LANGS[lang]
        self.lang = lang
        self.seed = seed
        self.families = families or list(tasks.FAMILIES)
        self.rng = random.Random(seed)

    # --- pretraining text ---------------------------------------------------
    def story(self, rng: random.Random) -> str:
        p = self.pack
        n = rng.randint(2, 5)
        people = rng.sample(p["people"], 2)
        out = []
        for _ in range(n):
            out.append(
                rng.choice(p["story"]).format(
                    p=people[0], q=people[1], l=rng.choice(p["places"]), o=rng.choice(p["objects"]),
                    v=rng.choice(p["verbs_past"]), w=rng.choice(p["weather"]), t=rng.choice(p["times"]),
                )
            )
        return ("" if not p["spaces"] else " ").join(out)

    def record(self, rng: random.Random) -> str:
        p = self.pack
        n = rng.randint(3, 6)
        lines = []
        for _ in range(n):
            lines.append(
                rng.choice(p["fact"]).format(
                    p=rng.choice(p["people"]), q=rng.choice(p["people"]), l=rng.choice(p["places"]),
                    o=rng.choice(p["objects"]), n=rng.randint(2, 40), u=rng.choice(p["units"]),
                )
            )
        return "\n".join(lines)

    def solved_task(self, rng: random.Random, difficulty: Optional[int] = None) -> dict:
        family = rng.choice(self.families)
        d = rng.choice([0, 1, 1, 2, 2, 3]) if difficulty is None else difficulty
        return tasks.generate(family, rng, self.pack, d)

    def worked_example(self, rng: random.Random) -> str:
        t = self.solved_task(rng)
        body = "\n".join(t["steps"])
        return f"{t['question']}\n{body}\n{self.pack['answer_prefix']} {t['answer']}"

    def documents(self, n: int, mix: Optional[dict[str, float]] = None, seed: Optional[int] = None) -> Iterator[dict[str, Any]]:
        """Pretraining documents. The default mix teaches fluency, facts and worked
        problems together, so the base model can already attempt a task before SFT."""
        mix = mix or {"story": 0.55, "record": 0.15, "task": 0.30}
        rng = random.Random(self.seed if seed is None else seed)
        kinds = list(mix)
        weights = [mix[k] for k in kinds]
        for i in range(n):
            kind = rng.choices(kinds, weights)[0]
            text = self.story(rng) if kind == "story" else self.record(rng) if kind == "record" else self.worked_example(rng)
            yield {"text": text, "kind": kind, "id": i}

    # --- instruction data ---------------------------------------------------
    def instructions(self, n: int, with_steps: bool = True, difficulty: Optional[int] = None, seed: Optional[int] = None) -> Iterator[dict[str, Any]]:
        """{prompt, answer, steps, family} — the prompt is the question alone; the
        target is the reasoning plus the marked answer (or just the answer)."""
        rng = random.Random((self.seed + 1000) if seed is None else seed)
        for i in range(n):
            t = self.solved_task(rng, difficulty)
            body = ("\n".join(t["steps"]) + "\n") if with_steps else ""
            yield {
                "id": i,
                "family": t["family"],
                "difficulty": t["difficulty"],
                "prompt": t["question"],
                "target": f"{body}{self.pack['answer_prefix']} {t['answer']}",
                "answer": t["answer"],
                "steps": t["steps"],
            }

    # --- evaluation sets ----------------------------------------------------
    def eval_set(self, n: int, seed: int = 999_983, difficulty: Optional[int] = None, families: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """A held-out set. The seed is far from the training seeds on purpose:
        the same generator, never the same problems."""
        rng = random.Random(seed)
        fams = families or self.families
        out = []
        for i in range(n):
            family = fams[i % len(fams)]
            d = rng.choice([0, 1, 2, 3]) if difficulty is None else difficulty
            t = tasks.generate(family, rng, self.pack, d)
            out.append({"id": i, "family": t["family"], "difficulty": t["difficulty"], "prompt": t["question"], "answer": t["answer"], "steps": t["steps"]})
        return out

    def extract(self, text: str) -> Optional[str]:
        return extract_answer(text, self.pack)

    def grade(self, text: str, gold: str) -> bool:
        return check_answer(self.extract(text), gold)

    @property
    def answer_prefix(self) -> str:
        return self.pack["answer_prefix"]

    @property
    def system_prompt(self) -> str:
        return self.pack["instruction_system"]


def multiple_choice(world, item: dict, rng, n_options: int = 4) -> dict:
    """Turn a generated question into a multiple-choice item.

    The distractors come from the same family, so they are the same kind of thing and
    the same order of magnitude — which is what makes a benchmark item hard rather
    than a vocabulary test."""
    gold = str(item["answer"])
    distractors: list[str] = []
    guard = 0
    while len(distractors) < n_options - 1 and guard < 200:
        guard += 1
        other = str(tasks.generate(item["family"], rng, world.pack, item.get("difficulty", 1))["answer"])
        if other != gold and other not in distractors:
            distractors.append(other)
    # numeric answers also get near-misses, which is where a model that half-computes
    # the answer gets caught
    if gold.lstrip("-").isdigit() and len(distractors) >= 2:
        near = str(int(gold) + rng.choice([-10, -2, -1, 1, 2, 10]))
        if near != gold:
            distractors[-1] = near
    options = distractors[: n_options - 1] + [gold]
    rng.shuffle(options)
    return {
        "question": item["prompt"],
        "options": options,
        "answer": options.index(gold),
        "family": item["family"],
        "difficulty": item.get("difficulty", 1),
        "gold_text": gold,
    }
