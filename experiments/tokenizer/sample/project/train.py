"""Train the project tokenizer on the sample pools and print statistics.

    python project/train.py --vocab-size 2000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer import Tokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vocab-size", type=int, default=2000)
ap.add_argument("--data", default=None, help="folder of .txt files (default: shipped pools)")
args = ap.parse_args()

materials = Path(os.environ.get("ATELIER_MATERIALS", "materials"))
experiments = Path(os.environ.get("ATELIER_EXPERIMENTS", Path(__file__).resolve().parents[3]))
candidates = [Path(args.data)] if args.data else [materials / "datasets/tokenizer/pools", experiments / "tokenizer/data/pools"]
data_dir = next((c for c in candidates if c.exists()), None)
if data_dir is None:
    raise SystemExit("No data folder found; pass --data")
texts = [ln.strip() for f in sorted(data_dir.rglob("*.txt")) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
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
stats = {"vocab_size": tok.vocab_size, "train_sec": round(train_s, 3), "holdout_docs": len(hold), "chars_per_token": round(chars / max(1, n_tokens), 3), "round_trip": ok, "encode_chars_per_sec": round(chars / max(1e-9, enc_s))}
print(json.dumps(stats, indent=2))
Path("outputs").mkdir(exist_ok=True)
Path("outputs/stats.json").write_text(json.dumps(stats, indent=2))
