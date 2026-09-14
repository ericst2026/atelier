"""Apply your filter and report what it keeps.  python project/run.py --limit 40000"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atelier_nlp import hf  # noqa: E402
from filters import keep  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=40000)
args = ap.parse_args()

raw = hf.dataset_split("curation", "c4-raw", "train", limit=args.limit)
good = hf.dataset_split("curation", "fineweb-edu", "train", limit=min(args.limit, 10000))
kept = [d for d in raw if keep(d)]
kept_good = sum(1 for d in good if keep(d))
print(f"raw:      kept {len(kept):,} of {len(raw):,}  ({len(kept) / len(raw):.1%})")
print(f"filtered: kept {kept_good:,} of {len(good):,}  ({kept_good / len(good):.1%})  ← should be high")
Path("outputs").mkdir(exist_ok=True)
hf.write_jsonl("outputs/kept.jsonl", kept)
print(f"{sum(len(d['text'] or '') for d in kept):,} characters written to outputs/kept.jsonl")
