"""Step 1 — build a corpus from the selected sources (optionally with planted duplicates)."""
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from atelier_sdk import Result, hist, inputs, params, parse_args, progress, write_jsonl
from atelier_nlp.text import char_classes, words

parse_args()
P = params({"sources": ["pools"], "max_docs": 2000, "split": "auto", "min_chars": 20, "exact_dup_rate": 0.1, "near_dup_rate": 0.1, "seed": 7})
I = inputs()
rng = random.Random(int(P["seed"]))
materials = Path(I.get("materials_dir") or os.environ.get("ATELIER_MATERIALS", "materials"))
experiment_dir = Path(I.get("experiment_dir") or Path(__file__).resolve().parents[1])
workspace = Path(I.get("workspace_dir") or ".")

SOURCE_PATHS = {
    "pools": [materials / "datasets/tokenizer/pools", experiment_dir / "data/pools"],
    "wikitext": [materials / "datasets/tokenizer/wikitext-103-raw"],
    "code": [materials / "datasets/tokenizer/code"],
    "multilingual": [materials / "datasets/tokenizer/multilingual"],
    "workspace": [workspace / "data"],
}
CODE_SUFFIXES = (".py", ".js", ".go", ".rs", ".sql", ".sh", ".c", ".java")


def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".txt", ".raw", ".md", ".jsonl") + CODE_SUFFIXES and "checksum" not in p.name:
            yield p


def iter_docs(path: Path, split: str):
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                t = d.get("text") or d.get("content") or ""
                if t:
                    yield t, d.get("lang") or d.get("language")
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if split == "whole" or path.suffix.lower() in CODE_SUFFIXES:
        yield text, None
    elif split == "line" or (split == "auto" and not re.search(r"\n\s*\n", text)):
        for ln in text.splitlines():
            yield ln, None
    else:
        for para in re.split(r"\n\s*\n", text):
            yield para.strip(), None


docs: list[dict] = []
limit = int(P["max_docs"])
available = []
for src in P["sources"]:
    roots = [r for r in SOURCE_PATHS.get(src, []) if r.exists()]
    if not roots:
        print(f"[corpus] source {src!r} not found on this server, skipped")
        continue
    available.append(src)
    n_before = len(docs)
    for f in iter_files(roots[0]):
        for text, lang in iter_docs(f, P["split"]):
            text = text.strip()
            if len(text) < int(P["min_chars"]):
                continue
            docs.append({"text": text, "source": src, "lang": lang, "file": f.name, "truth": "unique"})
            if len(docs) - n_before >= limit:
                break
        if len(docs) - n_before >= limit:
            break
    print(f"[corpus] {src}: {len(docs) - n_before} documents")
    progress(30 * len(available) / max(1, len(P["sources"])), f"read {src}")

if not docs:
    raise SystemExit("No documents found. Check the materials folder or upload files to data/ in your project.")

rng.shuffle(docs)
docs = docs[:limit]


def perturb(text: str, other: str) -> str:
    ws = text.split(" ")
    op = rng.randrange(8)
    if op == 0 and len(ws) > 3:
        del ws[rng.randrange(len(ws))]
    elif op == 1 and len(ws) > 3:
        i = rng.randrange(len(ws) - 1)
        ws[i], ws[i + 1] = ws[i + 1], ws[i]
    elif op == 2 and len(ws) > 1:
        i = rng.randrange(len(ws))
        ws.insert(i, ws[i])
    elif op == 3:
        i = rng.randrange(len(ws))
        ws[i] = ws[i].upper() if ws[i].islower() else ws[i].lower()
    elif op == 4:
        i = rng.randrange(len(ws))
        w = ws[i]
        if len(w) > 3:
            j = rng.randrange(len(w) - 1)
            ws[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    elif op == 5:
        i = rng.randrange(len(ws))
        w = ws[i]
        if len(w) > 2:
            j = rng.randrange(len(w))
            ws[i] = w[:j] + w[j + 1 :]
    elif op == 6:
        text2 = " ".join(ws)
        text2 = text2.replace(".", ",", 1) if "." in text2 else text2 + "."
        ws = text2.split(" ")
    else:
        frag = other.split(" ")[: max(1, min(6, len(other.split(" ")) // 3))]
        ws = (frag + ws) if rng.random() < 0.5 else (ws + frag)
    return " ".join(ws)


n_exact = int(round(len(docs) * float(P["exact_dup_rate"])))
n_near = int(round(len(docs) * float(P["near_dup_rate"])))
planted = []
for _ in range(n_exact):
    src = rng.choice(docs)
    planted.append({**src, "truth": "exact", "dup_of": docs.index(src)})
for _ in range(n_near):
    src = rng.choice(docs)
    other = rng.choice(docs)["text"]
    t = src["text"]
    for _ in range(rng.randint(1, 3)):
        t = perturb(t, other)
    planted.append({**src, "text": t, "truth": "near", "dup_of": docs.index(src)})
for i, d in enumerate(docs):
    d["id"] = i
for j, d in enumerate(planted):
    d["id"] = len(docs) + j
all_docs = docs + planted
rng.shuffle(all_docs)
progress(70, "planting duplicates")

out = Path(os.environ.get("ATELIER_RUN_DIR", ".")) / "corpus.jsonl"
write_jsonl(out, all_docs)

texts = [d["text"] for d in all_docs]
lengths = [len(t) for t in texts]
n_words = sum(len(words(t)) for t in texts)
classes = char_classes(texts)
by_source = Counter(d["source"] for d in all_docs)
distinct_chars = len(set("".join(texts)))
progress(90, "writing result")

R = Result()
R.metric("docs", "Documents", len(all_docs), "int", "raw", help=f"{len(docs)} unique + {n_exact} exact + {n_near} near planted")
R.metric("chars", "Characters", sum(lengths), "int", "raw")
R.metric("words", "Words", n_words, "int", "raw")
R.metric("distinct_chars", "Distinct characters", distinct_chars, "int", "sky")
R.chart("lengths", "Document lengths", hist(lengths, bins=24, log=max(lengths) / max(1, min(lengths)) > 50), "bin", [{"key": "count", "label": "Documents", "color": "raw"}], "bar", note="Characters per document; a log-spaced axis is used when lengths span more than two orders of magnitude.")
R.chart("classes", "Characters by class", [{"cls": k, "count": v} for k, v in classes.items()], "cls", [{"key": "count", "label": "Characters", "color": "sky"}], "bar")
R.chart("sources", "Documents by source", [{"source": k, "count": v} for k, v in by_source.most_common()], "source", [{"key": "count", "label": "Documents", "color": "raw"}], "bar")
if planted:
    R.chart("truth", "Planted duplicates", [{"truth": k, "count": v} for k, v in Counter(d["truth"] for d in all_docs).items()], "truth", [{"key": "count", "label": "Documents", "color": "dup"}], "bar")
R.table("preview", "Preview", [{"key": "id", "label": "#"}, {"key": "source", "label": "Source"}, {"key": "truth", "label": "Planted"}, {"key": "text", "label": "Text"}], [{"id": d["id"], "source": d["source"], "truth": "" if d["truth"] == "unique" else d["truth"], "text": d["text"][:160]} for d in all_docs[:40]])
R.artifact(out, "corpus.jsonl")
R.output("corpus", str(out)).output("sources", available).output("docs", len(all_docs))
R.note(f"Sources used: {', '.join(available)}. Planted {n_exact} exact and {n_near} near duplicates so the dedup step has known targets.")
R.save()
