"""Fast local sanity checks (a subset of what the grader runs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))
from tokenizer import Tokenizer  # noqa: E402

samples = ["The lighthouse keeper logged the tide at dawn.", "print(f\"{x:.2f}\")", "東京の朝は静かだ。", "  leading spaces and\ttabs\nnewlines  "]
tok = Tokenizer()
tok.train(samples * 10, 300)
assert tok.vocab_size <= 300, "vocab_size must not exceed the target"
for s in samples:
    ids = tok.encode(s)
    assert all(isinstance(i, int) for i in ids), "encode must return ints"
    assert tok.decode(ids) == s, f"round trip failed for {s!r}"
print("quick check passed: vocab", tok.vocab_size)
