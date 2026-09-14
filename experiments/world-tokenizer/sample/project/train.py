"""Train the project tokenizer on generated text and print what the grader measures.

    python project/train.py --vocab-size 4096 --lang en
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atelier_world import World  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vocab-size", type=int, default=4096)
ap.add_argument("--lang", default="en", choices=["en", "ja"])
ap.add_argument("--docs", type=int, default=8000)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

world = World(lang=args.lang, seed=args.seed)
texts = [d["text"] for d in world.documents(args.docs)]
split = int(len(texts) * 0.9)
train, hold = texts[:split], texts[split:]

tok = Tokenizer()
t0 = time.time()
tok.train(train, args.vocab_size)
train_s = time.time() - t0

chars = sum(len(t) for t in hold)
t0 = time.time()
n_tokens = sum(len(tok.encode(t)) for t in hold)
enc_s = time.time() - t0
ok = all(tok.decode(tok.encode(t)) == t for t in hold)
rng = random.Random(1)
numbers = [str(rng.randint(0, 999)) for _ in range(200)]
one_piece = sum(1 for n in numbers if len(tok.encode(n)) == 1) / len(numbers)

stats = {
    "vocab_size": tok.vocab_size,
    "train_sec": round(train_s, 2),
    "chars_per_token": round(chars / max(1, n_tokens), 3),
    "round_trip": ok,
    "numbers_as_one_token": round(one_piece, 3),
    "encode_chars_per_sec": round(chars / max(1e-9, enc_s)),
}
print(json.dumps(stats, indent=2, ensure_ascii=False))
Path("outputs").mkdir(exist_ok=True)
Path("outputs/stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
