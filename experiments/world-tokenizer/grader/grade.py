"""Grader: correctness, compression against the reference, digit handling, speed."""
import importlib.util
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress
from atelier_nlp import bpe
from atelier_world import World

VOCAB = 4096
TRAIN_LIMIT_SEC = 180
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
R = Result()
tests, score = [], 0.0


def test(name, passed, detail, pts=0.0, earned=0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{pts:.0f}"})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


class Timeout(Exception):
    pass


def with_timeout(seconds, fn, *a, **kw):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Timeout()))
    signal.alarm(seconds)
    try:
        return fn(*a, **kw)
    finally:
        signal.alarm(0)


# a seed the guided steps never use, so nobody can tune to these documents
world = World(lang=os.environ.get("ATELIER_GRADER_LANG", "en"), seed=523_711)
texts = [d["text"] for d in world.documents(12000)]
split = int(len(texts) * 0.85)
train, hold = texts[:split], texts[split:]
progress(5, f"{len(train)} training / {len(hold)} held-out documents")

Tok = None
try:
    spec = importlib.util.spec_from_file_location("student_tokenizer", project / "project" / "tokenizer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    spec.loader.exec_module(mod)
    Tok = mod.Tokenizer
    missing = [m for m in ("train", "encode", "decode") if not callable(getattr(Tok, m, None))]
    if missing:
        test("interface", False, f"missing methods: {missing}", 10, 0)
        Tok = None
    else:
        test("interface", True, "Tokenizer with train/encode/decode found", 10, 10)
except Exception as exc:
    test("interface", False, f"{type(exc).__name__}: {exc}", 10, 0)
    traceback.print_exc()

tok = None
if Tok is not None:
    try:
        tok = Tok()
        t0 = time.time()
        with_timeout(TRAIN_LIMIT_SEC, tok.train, list(train), VOCAB)
        train_s = time.time() - t0
        vs = int(getattr(tok, "vocab_size", VOCAB))
        ok = vs <= VOCAB
        test("train", ok, f"trained in {train_s:.1f}s, vocab_size={vs}" + ("" if ok else f" (over {VOCAB})"), 10, 10 if ok else 3)
        R.metric("train_sec", "Training time", train_s, "num", "hold")
    except Timeout:
        test("train", False, f"training took longer than {TRAIN_LIMIT_SEC}s", 10, 0)
        tok = None
    except Exception as exc:
        test("train", False, f"{type(exc).__name__}: {exc}", 10, 0)
        traceback.print_exc()
        tok = None
progress(40, "training done")

cpt = None
if tok is not None:
    try:
        ok_n, n_tokens, chars = 0, 0, 0
        t0 = time.time()
        for t in hold:
            ids = tok.encode(t)
            n_tokens += len(ids)
            chars += len(t)
            if tok.decode(ids) == t:
                ok_n += 1
        enc_s = time.time() - t0
        rate = ok_n / max(1, len(hold))
        test("round_trip", rate == 1.0, f"{ok_n}/{len(hold)} documents decode exactly", 25, 25 * rate)
        R.metric("round_trip_rate", "Round-trip rate", rate, "pct", "kept" if rate == 1 else "dup")
        speed = chars / max(1e-9, enc_s)
        test("speed", speed >= 50_000, f"{speed:,.0f} chars/s", 10, 10 * min(1.0, max(0.0, (math.log10(max(speed, 1)) - 3.5) / 2.5)))
        R.metric("encode_speed", "Encoding speed", speed, "num", "sky")
        cpt = chars / max(1, n_tokens)
        rng = random.Random(17)
        numbers = [str(rng.randint(0, 999)) for _ in range(300)]
        consistent = sum(1 for n in numbers if len(tok.encode(n)) in (1, len(n))) / len(numbers)
        test("digits", consistent >= 0.9, f"{consistent:.0%} of numbers encode consistently (one token, or one per digit)", 15, 15 * consistent)
        R.metric("digit_consistency", "Digit consistency", consistent, "pct", "raw")
    except Exception as exc:
        test("round_trip", False, f"{type(exc).__name__}: {exc}", 25, 0)
        traceback.print_exc()

progress(70, "training the reference for comparison")
ref = bpe.train(bpe.count_words(train), "byte", VOCAB, 2)
ref_cpt = bpe.evaluate(ref, hold, curve_vocabs=[VOCAB])["chars_per_token"]
R.metric("reference_cpt", "Reference chars/token", ref_cpt, "num", "hold", help=f"byte-level BPE, vocabulary {VOCAB}")
if cpt is not None:
    ratio = cpt / max(1e-9, ref_cpt)
    test("compression", ratio >= 0.95, f"{cpt:.3f} chars/token against the reference {ref_cpt:.3f} (ratio {ratio:.2f})", 40, 40 * min(1.0, max(0.0, (ratio - 0.4) / 0.6)))
    R.metric("chars_per_token", "Chars per token", cpt, "num", "kept" if ratio >= 1 else "raw")
    R.metric("compression_ratio", "Against the reference", ratio, "num", "sky")
else:
    test("compression", False, "skipped: no working tokenizer", 40, 0)

score = round(min(100.0, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup", help="out of 100")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.chart("bars", "Where the points came from", [{"check": t["name"], "earned": float(t["points"].split("/")[0]), "max": float(t["points"].split("/")[1])} for t in tests], "check", [{"key": "earned", "label": "Earned", "color": "kept"}, {"key": "max", "label": "Available", "color": "hold"}], "bar")
R.output("score", score)
progress(100, f"score {score}")
R.save()
