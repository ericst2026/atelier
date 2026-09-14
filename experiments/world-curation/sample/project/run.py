"""Score your filter locally.  python project/run.py --docs 20000"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atelier_world import World, make_corpus, score_filter  # noqa: E402
from filters import keep  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--docs", type=int, default=20000)
ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
args = ap.parse_args()

world = World(lang=args.lang, seed=args.seed)
docs = make_corpus(world, args.docs, seed=args.seed)
kept = [keep(d) for d in docs]
s = score_filter(docs, kept)
print(f"precision {s['precision']:.3f}  recall {s['recall']:.3f}  F1 {s['f1']:.3f}")
print(f"kept {s['kept']:,} of {len(docs):,} documents")
print("kept by kind:")
for k, v in s["by_kind"].items():
    target = "keep" if k == "clean" else "drop"
    print(f"  {k:14s} {v:6.1%}   (should {target})")
