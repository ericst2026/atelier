"""Grader for the tokenizer project.

    python grader/grade.py --project <dir> --run-dir <dir>

Trains the student's Tokenizer on a fixed corpus and scores four things:
interface & round-trip correctness, compression against the reference BPE at the
same vocabulary size, and encoding speed. Writes result.json with `score`."""
import importlib.util
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

from atelier_sdk import Result, parse_args, progress, read_jsonl
from atelier_nlp import bpe

VOCAB = 2000
TRAIN_LIMIT_SEC = 120
ENCODE_LIMIT_SEC = 60
args = parse_args()
project = Path(args.project or os.environ.get("ATELIER_WORKSPACE", ".")).resolve()
materials = Path(os.environ.get("ATELIER_MATERIALS", "materials"))
experiment_dir = Path(__file__).resolve().parents[1]


def load_texts() -> list[str]:
    texts = []
    pools = next((p for p in (materials / "datasets/tokenizer/pools", experiment_dir / "data/pools") if p.exists()), None)
    if pools:
        for f in sorted(pools.rglob("*.txt")):
            texts += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    wiki = materials / "datasets/tokenizer/wikitext-103-raw"
    if wiki.exists():
        for f in sorted(wiki.rglob("*")):
            if f.is_file() and f.suffix in (".raw", ".txt"):
                for para in f.read_text(encoding="utf-8", errors="replace").split("\n\n"):
                    para = para.strip()
                    if len(para) > 80:
                        texts.append(para)
                    if len(texts) > 4000:
                        break
            if len(texts) > 4000:
                break
    rng = random.Random(11)
    rng.shuffle(texts)
    return texts


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def with_timeout(seconds: int, fn, *a, **kw):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return fn(*a, **kw)
    finally:
        signal.alarm(0)


R = Result()
tests: list[dict] = []
score = 0.0


def test(name: str, passed: bool, detail: str = "", points: float = 0.0, earned: float = 0.0):
    global score
    score += earned
    tests.append({"name": name, "passed": passed, "detail": detail, "points": f"{earned:.1f}/{points:.0f}" if points else ""})
    print(f"[{'PASS' if passed else 'FAIL'}] {name} — {detail}")


texts = load_texts()
split = int(len(texts) * 0.85)
train, hold = texts[:split], texts[split:]
progress(5, f"{len(train)} training / {len(hold)} held-out texts")

# 1. interface
Tok = None
try:
    entry = project / "project" / "tokenizer.py"
    spec = importlib.util.spec_from_file_location("student_tokenizer", entry)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(project / "project"))
    sys.path.insert(0, str(project))
    spec.loader.exec_module(mod)
    Tok = getattr(mod, "Tokenizer")
    missing = [m for m in ("train", "encode", "decode") if not callable(getattr(Tok, m, None))]
    if missing:
        test("interface", False, f"missing methods: {missing}", 15, 0)
        Tok = None
    else:
        test("interface", True, "Tokenizer with train/encode/decode found", 15, 15)
except Exception as exc:
    test("interface", False, f"{type(exc).__name__}: {exc}", 15, 0)
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
        test("train", ok, f"trained in {train_s:.1f}s, vocab_size={vs}" + ("" if ok else f" (> {VOCAB})"), 10, 10 if ok else 4)
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
    # 3. round trip + speed
    ok_n, n_tokens, chars = 0, 0, 0
    t0 = time.time()
    try:
        for t in hold:
            ids = with_timeout(ENCODE_LIMIT_SEC, tok.encode, t)
            n_tokens += len(ids)
            chars += len(t)
            if tok.decode(ids) == t:
                ok_n += 1
        enc_s = time.time() - t0
        rate = ok_n / max(1, len(hold))
        test("round_trip", rate == 1.0, f"{ok_n}/{len(hold)} held-out texts decode exactly", 25, 25 * rate)
        R.metric("round_trip_rate", "Round-trip rate", rate, "pct", "kept" if rate == 1 else "dup")
        speed = chars / max(1e-9, enc_s)
        sp_pts = 10 * min(1.0, max(0.0, (math.log10(max(speed, 1)) - 3) / 3))  # 1k chars/s → 0, 1M chars/s → 10
        test("speed", speed >= 20_000, f"{speed:,.0f} chars/s", 10, sp_pts)
        R.metric("encode_speed", "Encoding speed", speed, "num", "sky", help="characters per second on held-out text")
        cpt = chars / max(1, n_tokens)
    except Timeout:
        test("round_trip", False, f"encode took longer than {ENCODE_LIMIT_SEC}s on one text", 25, 0)
    except Exception as exc:
        test("round_trip", False, f"{type(exc).__name__}: {exc}", 25, 0)
        traceback.print_exc()
progress(70, "computing the reference")

# 4. compression against reference BPE at the same vocab size
ref = bpe.train(bpe.count_words(train), "byte", VOCAB, 2)
ref_eval = bpe.evaluate(ref, hold, curve_vocabs=[VOCAB])
ref_cpt = ref_eval["chars_per_token"]
R.metric("reference_cpt", "Reference chars/token", ref_cpt, "num", "hold", help=f"byte-level BPE, vocab {VOCAB}")
if cpt is not None:
    ratio = cpt / max(1e-9, ref_cpt)
    pts = 40 * min(1.0, max(0.0, (ratio - 0.4) / 0.6)) if ratio < 1 else 40
    test("compression", ratio >= 0.95, f"{cpt:.3f} chars/token vs reference {ref_cpt:.3f} (ratio {ratio:.2f})", 40, pts)
    R.metric("chars_per_token", "Chars per token", cpt, "num", "kept" if ratio >= 1 else "raw")
    R.metric("compression_ratio", "vs reference", ratio, "num", "sky")
else:
    test("compression", False, "skipped: no working tokenizer", 40, 0)

score = round(min(100.0, score), 1)
R.metric("score", "Score", score, "num", "kept" if score >= 70 else "dup", help="out of 100")
R.table("tests", "Checks", [{"key": "name", "label": "Check"}, {"key": "passed", "label": "Passed"}, {"key": "points", "label": "Points"}, {"key": "detail", "label": "Detail"}], tests)
R.chart("bars", "Where the points came from", [{"check": t["name"], "earned": float(t["points"].split("/")[0]) if t["points"] else 0, "max": float(t["points"].split("/")[1]) if t["points"] else 0} for t in tests], "check", [{"key": "earned", "label": "Earned", "color": "kept"}, {"key": "max", "label": "Available", "color": "hold"}], "bar")
R.output("score", score)
progress(100, f"score {score}")
R.save()
sys.exit(0)
