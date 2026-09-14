"""Step 1 — the corpus.

Three sources: generated from the world (no downloads, reproducible from the seed),
a folder of your own text under materials/, or text pasted into the form."""
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, params, parse_args, progress, write_jsonl
from atelier_world import World
from atelier_nlp.text import char_classes

parse_args()
P = params({"source": "generated", "corpus_path": "", "pasted_text": "", "lang": "en", "docs": 20000, "story_share": 0.55, "record_share": 0.15, "task_share": 0.30, "dup_rate": 0.08, "seed": 7})
run_dir = Path(os.environ.get("ATELIER_RUN_DIR", "."))
source = str(P["source"] or "generated")


def read_folder(rel: str, limit: int) -> list[dict]:
    """Every .txt, .md and .jsonl under a folder in materials/. One document per file,
    or per line for JSONL — whichever the file is."""
    root = Path(os.environ.get("ATELIER_MATERIALS", "/srv/atelier/materials")) / rel.strip().lstrip("/")
    if not root.exists():
        raise SystemExit(f"No such folder: {root}. Put your text under materials/ and give the path relative to it, for example corpora/our-reports.")
    docs, files = [], 0
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".txt", ".md", ".jsonl", ".json"):
            continue
        files += 1
        try:
            if f.suffix.lower() in (".jsonl", ".json"):
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    text = row.get("text") or row.get("content") or row.get("body") or ""
                    if isinstance(text, str) and text.strip():
                        docs.append({"text": text.strip(), "kind": f.stem, "id": len(docs), "truth": "unknown", "file": f.name})
                    if len(docs) >= limit:
                        break
            else:
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    docs.append({"text": text, "kind": f.suffix.lstrip("."), "id": len(docs), "truth": "unknown", "file": f.name})
        except OSError:
            continue
        if len(docs) >= limit:
            break
    if not docs:
        raise SystemExit(f"Found {files} file(s) under {root} but no readable text in them. Expected .txt, .md, or .jsonl with a \"text\" field.")
    return docs[:limit]


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

if source == "folder":
    progress(10, f"reading {P['corpus_path']!r} from materials")
    docs = read_folder(str(P["corpus_path"]), n)
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
    for i, d in enumerate(world.documents(n, mix)):
        docs.append({**d, "truth": "unique"})
        if i % 2000 == 0:
            progress(70 * i / n, f"{i:,} documents")
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
R.chart("kinds", "Documents by kind", [{"kind": k, "count": v} for k, v in kinds.items()], "kind", [{"key": "count", "label": "Documents", "color": "raw"}], "bar")
R.chart("lengths", "Document lengths", hist(lengths, bins=24), "bin", [{"key": "count", "label": "Documents", "color": "kept"}], "bar")
R.chart("classes", "Characters by class", [{"cls": k, "count": v} for k, v in classes.items() if v], "cls", [{"key": "count", "label": "Characters", "color": "sky"}], "bar", note="Japanese fills the cjk bucket and leaves spaces nearly empty — the reason its tokenizer behaves differently.")
R.table("preview", "Preview", [{"key": "kind", "label": "Kind"}, {"key": "text", "label": "Text"}], [{"kind": d["kind"], "text": d["text"][:400]} for d in docs[:25]])
R.artifact(out, "corpus.jsonl")
R.output("corpus", str(out)).output("lang", P["lang"]).output("docs", len(docs))
if source == "generated":
    R.note(f"Language: {world.pack['name']}. Mix: " + ", ".join(f"{k} {v:.0%}" for k, v in mix.items()) + f". Seed {P['seed']} regenerates exactly this corpus.")
else:
    where = f"materials/{P['corpus_path']}" if source == "folder" else "the text you pasted"
    R.note(f"Your own text, from {where}. No duplicates were planted — whatever the next step finds was already there, which is the more interesting case. Everything downstream (pretraining, fine-tuning, the rest) uses the vocabulary this produces, so this is where the course starts working on your data rather than ours.")
R.save()
