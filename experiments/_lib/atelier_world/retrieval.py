"""A searchable library, generated.

Each document is a short profile of one entity with several facts in it. Each query
asks about exactly one fact, so the gold document is known by construction — no
annotation, and no ambiguity about what counts as a correct retrieval.

The distractors are the point. Every entity shares its place, its objects or its
trade with several others, so a query that matches on words alone retrieves the
wrong profile. Lexical search gets some of these; only something that represents
meaning gets the rest."""
import random
from typing import Any, Optional


FIELDS = ("number", "companion", "name", "place", "object", "trade")


def _profile(pack: dict, rng: random.Random, name: str, shared: dict) -> dict:
    return {
        "name": name,
        "place": shared["place"],
        "object": shared["object"],
        "trade": rng.choice(pack["verbs_past"]),
        "number": rng.randint(100, 999),
        "companion": shared["companion"],
        "time": rng.choice(pack["times"]),
    }


def _render(pack: dict, p: dict) -> str:
    sp = " " if pack["spaces"] else ""
    if pack["code"] == "ja":
        lines = [
            f"{p['name']}は{p['place']}で働いている。",
            f"{p['name']}は{p['object']}を持っていて、その番号は{p['number']}である。",
            f"{p['time']}、{p['name']}は{p['companion']}と一緒に{p['object']}を{p['trade']}。",
        ]
    else:
        lines = [
            f"{p['name']} works at {p['place']}.",
            f"{p['name']} keeps {p['object']}, numbered {p['number']}.",
            f"{p['time'].capitalize()}, {p['name']} {p['trade']} {p['object']} with {p['companion']}.",
        ]
    return sp.join(lines) if pack["spaces"] else "".join(lines)


def _question(pack: dict, p: dict, field: str) -> tuple[str, str]:
    """The query never names the entity and never reuses the document's wording for
    the detail that identifies it. Word overlap therefore points at the whole cluster,
    and something has to tell the members apart."""
    alt_verb = pack["verb_alt"].get(p["trade"], p["trade"])
    alt_time = pack["time_alt"].get(p["time"], p["time"])
    if pack["code"] == "ja":
        who = f"{p['place']}で、{alt_time}に{p['object']}を{alt_verb}人"
        table = {
            "number": (f"{who}の{p['object']}の番号は何ですか。", str(p["number"])),
            "companion": (f"{who}は誰と一緒でしたか。", p["companion"]),
            "name": (f"{who}の名前は何ですか。", p["name"]),
            "place": (f"{alt_time}に{p['object']}を{alt_verb}人はどこで働いていますか。", p["place"]),
            "object": (f"{p['place']}で{alt_time}に{alt_verb}のは何ですか。", p["object"]),
            "trade": (f"{p['place']}の{p['companion']}と一緒にいた人は{p['object']}をどうしましたか。", p["trade"]),
        }
    else:
        scene = f"At {p['place']}, someone {alt_verb} {p['object']} {alt_time}."
        table = {
            "number": (f"{scene} What number is on it?", str(p["number"])),
            "companion": (f"{scene} Who was with them?", p["companion"]),
            "name": (f"{scene} Who was that?", p["name"]),
            "place": (f"Someone {alt_verb} {p['object']} {alt_time}. Where do they work?", p["place"]),
            "object": (f"At {p['place']}, someone {alt_verb} something {alt_time}. What was it?", p["object"]),
            "trade": (f"At {p['place']}, someone was there with {p['companion']} and {p['object']}. What did they do with it?", p["trade"]),
        }
    return table[field]


def _question_old(pack: dict, p: dict, field: str) -> tuple[str, str]:
    if pack["code"] == "ja":
        table = {
            "place": (f"{p['name']}はどこで働いていますか。", p["place"]),
            "object": (f"{p['name']}は何を持っていますか。", p["object"]),
            "trade": (f"{p['name']}は{p['object']}をどうしましたか。", p["trade"]),
            "number": (f"{p['name']}の{p['object']}の番号は何ですか。", str(p["number"])),
            "companion": (f"{p['name']}は誰と一緒でしたか。", p["companion"]),
            "time": (f"{p['name']}はいつ{p['object']}を扱いましたか。", p["time"]),
        }
    else:
        table = {
            "place": (f"Where does {p['name']} work?", p["place"]),
            "object": (f"What does {p['name']} keep?", p["object"]),
            "trade": (f"What did {p['name']} do with {p['object']}?", p["trade"]),
            "number": (f"What number is on {p['name']}'s {p['object']}?", str(p["number"])),
            "companion": (f"Who was with {p['name']}?", p["companion"]),
            "time": (f"When did {p['name']} handle {p['object']}?", p["time"]),
        }
    return table[field]


class Library:
    """documents: [{id, name, text}] · queries: [{query, answer, gold_id, field}]"""

    def __init__(self, world, n_docs: int = 5000, seed: int = 4242, cluster_size: int = 4):
        pack = world.pack
        rng = random.Random(seed)
        self.world = world
        self.documents: list[dict[str, Any]] = []
        self.profiles: list[dict[str, Any]] = []
        used: set[str] = set()
        while len(self.profiles) < n_docs:
            # a cluster of entities that share a place, an object and a companion:
            # word overlap cannot tell them apart, only the name can
            # a cluster shares its place and its object, so those words identify four
            # documents rather than one; members differ in what they did and when
            shared = {
                "place": rng.choice(pack["places"]),
                "object": rng.choice(pack["objects"]),
                "companion": rng.choice(pack["people"]),
            }
            for _ in range(min(cluster_size, n_docs - len(self.profiles))):
                # two given names joined, so a few thousand entities all read as people
                name = _compose(pack, rng, used)
                used.add(name)
                p = _profile(pack, rng, name, shared)
                self.profiles.append(p)
                self.documents.append({"id": len(self.documents), "name": name, "text": _render(pack, p)})

    def queries(self, n: int, seed: int = 999_331, fields: Optional[list[str]] = None) -> list[dict[str, Any]]:
        rng = random.Random(seed)
        fields = fields or list(FIELDS)
        out = []
        for i in range(n):
            idx = rng.randrange(len(self.profiles))
            p = self.profiles[idx]
            field = fields[i % len(fields)]
            q, a = _question(self.world.pack, p, field)
            out.append({"id": i, "query": q, "answer": a, "gold_id": idx, "field": field, "name": p["name"]})
        return out

    def pairs(self, n: int, seed: int = 20_240_101, fields: Optional[list[str]] = None) -> list[dict[str, Any]]:
        """Training pairs: a query and the document that answers it. The seed is far
        from the evaluation seed, so the questions differ even though the library does not."""
        rows = self.queries(n, seed, fields)
        for r in rows:
            r["positive"] = self.documents[r["gold_id"]]["text"]
        return rows


def _compose(pack: dict, rng: random.Random, used: set) -> str:
    joiner = "・" if not pack["spaces"] else " "
    for _ in range(200):
        parts = rng.sample(pack["people"], 2)
        name = joiner.join(parts)
        if name not in used:
            return name
    for _ in range(2000):
        parts = rng.sample(pack["people"], 3)
        name = joiner.join(parts)
        if name not in used:
            return name
    raise RuntimeError("ran out of distinct names; reduce n_docs or enlarge the language pack")
