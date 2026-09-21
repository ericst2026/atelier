"""Step 1 — the corpus.

Three sources: generated from the world (no downloads, reproducible from the seed),
a prepared dataset of your own under materials/datasets, or text pasted into the form."""
import os
import random
import re
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, params, parse_args, progress, write_jsonl
from atelier_world import World
from atelier_world.prepared import read_documents
from atelier_nlp.text import char_classes

parse_args()
P = params({"source": "generated", "data_material": None, "pasted_text": "", "lang": "en", "docs": 20000, "story_share": 0.55, "record_share": 0.15, "task_share": 0.30, "dup_rate": 0.08, "seed": 7})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
source = str(P["source"] or "generated")
material = P.get("data_material")
if source == "folder":
    # runs saved before the picker typed a folder path; it is the same thing
    source, material = "prepared", material or P.get("corpus_path")


def read_pasted(raw: str, limit: int) -> list[dict]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw or "") if b.strip()]
    if not blocks:
        raise SystemExit("Nothing pasted. Put your text in the box, one document per blank-line-separated block.")
    return [{"text": b, "kind": "pasted", "id": i, "truth": "unknown"} for i, b in enumerate(blocks[:limit])]


total = float(P["story_share"]) + float(P["record_share"]) + float(P["task_share"])
world = World(lang=P["lang"], seed=int(P["seed"]))
n = int(P["docs"])
rng = random.Random(int(P["seed"]) + 1)
planted = 0

if source == "prepared":
    if not material:
        raise SystemExit("Choose a prepared dataset from the list, or switch the source back to generated.")
    progress(10, f"reading materials/{material}")
    docs = read_documents(str(material), limit=n)
    mix = {}
    progress(75, f"{len(docs):,} documents of your own")
elif source == "pasted":
    docs = read_pasted(str(P["pasted_text"]), n)
    mix = {}
    progress(75, f"{len(docs):,} pasted documents")
else:
    if total <= 0:
        raise SystemExit("At least one share must be above zero.")
    mix = {"story": float(P["story_share"]) / total, "record": float(P["record_share"]) / total, "task": float(P["task_share"]) / total}
    docs = []
    by_kind: dict = {}
    for i, d in enumerate(world.documents(n, mix)):
        docs.append({**d, "truth": "unique"})
        by_kind[d.get("kind", "other")] = by_kind.get(d.get("kind", "other"), 0) + 1
        if i % 2000 == 0:
            progress(70 * i / n, f"{i:,} documents", step=i + 1, x_label="documents written", **{f"documents__{k}": v for k, v in by_kind.items()})
    # duplicates are planted only in generated text, where we know there were none.
    # Real corpora have their own, which is what the next step is for.
    planted = int(n * float(P["dup_rate"]))
for _ in range(planted):
    src = rng.choice(docs[:n])
    text = src["text"]
    kind = "exact"
    if rng.random() < 0.5:
        units = text.split(" ") if world.pack["spaces"] else list(text)
        if len(units) > 4:
            i = rng.randrange(len(units) - 1)
            units[i], units[i + 1] = units[i + 1], units[i]
        text = (" " if world.pack["spaces"] else "").join(units)
        kind = "near"
    docs.append({"text": text, "kind": src["kind"], "id": len(docs), "truth": kind, "dup_of": src["id"]})
if source == "generated":
    rng.shuffle(docs)
progress(85, "writing the corpus")

out = run_dir / "corpus.jsonl"
write_jsonl(out, docs)
texts = [d["text"] for d in docs]
lengths = [len(t) for t in texts]
chars = sum(lengths)
classes = char_classes(texts[:5000])
kinds = Counter(d["kind"] for d in docs)

R = Result()
R.metric("docs", "Documents", len(docs), "int", "raw", help=f"{planted} duplicates planted")
R.metric("chars", "Characters", chars, "int", "raw")
R.metric("distinct_chars", "Distinct characters", len(set("".join(texts[:5000]))), "int", "sky", help="the alphabet the tokenizer must cover")
R.metric("mean_len", "Characters per document", chars / len(docs), "num", "sky")
R.chart("kinds", "Documents by kind", [{"kind": k, "count": v} for k, v in kinds.items()], "kind", [{"key": "count", "label": "Documents"}], "donut", note="The mix you asked for, as it came out.")
R.chart("lengths", "Document lengths", hist(lengths, bins=24), "bin", [{"key": "count", "label": "Documents", "color": "kept"}], "bar")
R.chart("classes", "Characters by class", [{"cls": k, "count": v} for k, v in classes.items() if v], "cls", [{"key": "count", "label": "Characters"}], "donut", note="Japanese fills the cjk bucket and leaves spaces nearly empty — the reason its tokenizer behaves differently.")
R.table("preview", "Preview", [{"key": "kind", "label": "Kind"}, {"key": "text", "label": "Text"}], [{"kind": d["kind"], "text": d["text"][:400]} for d in docs[:25]])
R.artifact(out, "corpus.jsonl")
R.output("corpus", str(out)).output("lang", P["lang"]).output("docs", len(docs)).output("data_source", source)
if source == "generated":
    R.note(f"Language: {world.pack['name']}. Mix: " + ", ".join(f"{k} {v:.0%}" for k, v in mix.items()) + f". Seed {P['seed']} regenerates exactly this corpus.")
else:
    where = f"materials/{material}" if source == "prepared" else "the text you pasted"
    R.note(f"Your own text, from {where}. No duplicates were planted — whatever the next step finds was already there, which is the more interesting case. Everything downstream (pretraining, fine-tuning, the rest) uses the vocabulary this produces, so this is where the course starts working on your data rather than ours.")
R.save()
