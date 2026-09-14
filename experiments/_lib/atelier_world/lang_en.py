"""English language pack. A pack is data only — copy it to add a language."""

PACK = {
    "code": "en",
    "name": "English",
    "spaces": True,
    "people": ["Mira", "Tomas", "Ada", "Bo", "Ines", "Karel", "Nadia", "Owen", "Petra", "Rafi", "Suki", "Vik", "Wren", "Yusuf", "Zara", "Lena", "Milo", "Orla", "Pavel", "Ruth", "Sanjay", "Tess", "Ulla", "Viktor", "Wanda", "Ximena", "Yara", "Zoltan", "Anders", "Beatrix"],
    "places": ["the harbour", "the workshop", "the orchard", "the library", "the lighthouse", "the market", "the bakery", "the boatyard", "the observatory", "the greenhouse", "the mill", "the archive", "the quarry", "the dairy", "the printworks", "the aviary"],
    "objects": ["a brass compass", "a coil of rope", "a paper lantern", "a tin whistle", "a folding chair", "a glass jar", "a wool blanket", "a copper kettle", "a leather satchel", "a wooden crate", "a clay pot", "a silver key", "a cotton sail", "a stack of maps", "a bag of walnuts", "a pocket watch"],
    "verbs_past": ["carried", "repaired", "counted", "sketched", "hid", "polished", "weighed", "sold", "borrowed", "measured", "packed", "traded"],
    "weather": ["a cold fog", "steady rain", "a dry wind", "bright sun", "low cloud", "a light frost"],
    "times": ["at dawn", "before noon", "in the afternoon", "at dusk", "after the tide turned", "on the first clear morning"],
    "units": ["metres", "crates", "coins", "baskets", "sacks", "jars"],
    "goods": ["apples", "candles", "nails", "ropes", "lamps", "bricks", "eggs", "pears", "hinges", "tiles"],
    # story templates; {p}=person {q}=other person {l}=place {o}=object {v}=verb {w}=weather {t}=time
    "story": [
        "{t} {p} walked to {l} and {v} {o}.",
        "There was {w} over {l}, so {p} stayed inside and {v} {o}.",
        "{p} met {q} at {l}. They talked about {o} until the light went.",
        "{p} kept {o} in {l} because {q} had asked for it.",
        "When {p} arrived at {l}, {q} had already {v} {o}.",
        "{t} the road to {l} was quiet, and {p} {v} {o} without hurry.",
        "{p} said that {o} belonged in {l}, and {q} agreed.",
        "Nobody at {l} had seen {o} since {p} {v} it {t}.",
        "{p} counted what was left in {l} and wrote the number down.",
        "After {w}, {p} and {q} carried {o} from {l} to the house.",
    ],
    "fact": [
        "{p} lives near {l}.",
        "{p} keeps {o} at {l}.",
        "{p} works at {l} with {q}.",
        "{p} has {n} {u}.",
    ],
    # task phrasing
    # A second way of saying the same thing. Documents use the plain form, queries use
    # this one, so a query and its answer share almost no content words — which is the
    # gap between lexical search and an embedder, and the whole point of the experiment.
    "verb_alt": {
        "carried": "hauled", "repaired": "mended", "counted": "tallied", "sketched": "drew",
        "hid": "concealed", "polished": "buffed", "weighed": "balanced", "sold": "parted with",
        "borrowed": "requisitioned", "measured": "sized up", "packed": "stowed", "traded": "swapped",
    },
    "time_alt": {
        "at dawn": "as the sun came up", "before noon": "in the late morning", "in the afternoon": "once the day had worn on",
        "at dusk": "as the light was going", "after the tide turned": "once the water began to fall back",
        "on the first clear morning": "the first day the sky was open",
    },
    "q_arith": ["What is {expr}?", "Work out {expr}.", "Calculate {expr}."],
    "q_count": ["How many {g} are there in total?", "Count the {g}.", "What is the total number of {g}?"],
    "q_sort": ["Sort these numbers from smallest to largest: {items}.", "Put these in order, smallest first: {items}."],
    "q_lookup": ["{facts}\nWhere does {p} keep {o}?", "{facts}\nWho works at {l}?"],
    "q_path": ["{p} starts at {l0} and goes {moves}. Where does {p} end up?"],
    "q_shop": ["{p} buys {n1} {g1} at {c1} coins each and {n2} {g2} at {c2} coins each. How much is that?"],
    "q_compare": ["{p} has {n1} {u} and {q} has {n2} {u}. How many more does {more} have?"],
    "q_seq": ["What number comes next: {items}?"],
    # reasoning-step phrasing
    "s_then": "Then {a} = {b}.",
    "s_first": "First {a} = {b}.",
    "s_total": "So the total is {b}.",
    "s_answer": "The answer is {b}.",
    "s_count": "{g}: {items}, which is {b}.",
    "s_move": "From {a} going {d} reaches {b}.",
    "s_cost": "{n} at {c} each is {b} coins.",
    "directions": {"north": "north", "south": "south", "east": "east", "west": "west"},
    "yes": "yes",
    "no": "no",
    "answer_prefix": "Answer:",
    "instruction_system": "Answer the question. Put the final answer on a line beginning with 'Answer:'.",
}
