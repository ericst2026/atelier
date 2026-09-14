"""Byte-pair encoding, written to be read.

Trainer keeps pair counts incrementally and picks the next merge from a lazy
max-heap, so training a few thousand merges on a million-word corpus takes
seconds, not minutes. The encoder records which merge ranks it applied to each
word; that trace lets us evaluate every intermediate vocabulary size without
retraining (see `evaluate`)."""
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .text import pretokenize

UNK = 0


def count_words(texts: Iterable[str], lowercase: bool = False) -> Counter:
    c: Counter = Counter()
    for t in texts:
        if lowercase:
            t = t.lower()
        c.update(pretokenize(t))
    return c


class BPEModel:
    """vocab: list of dicts {id, sym(bytes or str), display, kind, freq}; merges: list of (a, b, new_id, freq)."""

    def __init__(self, mode: str, lowercase: bool, base_size: int, vocab: list[dict[str, Any]], merges: list[tuple[int, int, int, int]], checkpoints: Optional[list[dict[str, Any]]] = None, train_stats: Optional[dict[str, Any]] = None):
        self.mode, self.lowercase, self.base_size = mode, lowercase, base_size
        self.vocab, self.merges = vocab, merges
        self.checkpoints = checkpoints or []
        self.train_stats = train_stats or {}
        self.rank: dict[tuple[int, int], tuple[int, int]] = {(a, b): (new, r) for r, (a, b, new, _f) in enumerate(merges)}
        self.base_id: dict[Any, int] = {}
        if mode == "char":
            for v in vocab[:base_size]:
                if v["id"] != UNK:
                    self.base_id[v["raw"]] = v["id"]
        self._cache: dict[str, tuple[list[int], list[int], int]] = {}

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"format": "atelier-bpe-1", "mode": self.mode, "lowercase": self.lowercase, "base_size": self.base_size, "vocab": self.vocab, "merges": [list(m) for m in self.merges], "checkpoints": self.checkpoints, "train_stats": self.train_stats}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "BPEModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["mode"], d["lowercase"], d["base_size"], d["vocab"], [tuple(m) for m in d["merges"]], d.get("checkpoints"), d.get("train_stats"))

    # -- encoding ---------------------------------------------------------
    def _units(self, word: str) -> list[int]:
        if self.mode == "byte":
            return [b + 1 for b in word.encode("utf-8")]
        return [self.base_id.get(ch, UNK) for ch in word]

    def encode_word(self, word: str) -> tuple[list[int], list[int], int]:
        """Returns (ids, applied merge ranks sorted, unk_units)."""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = self._units(word)
        unk = sum(1 for i in ids if i == UNK)
        trace: list[int] = []
        while len(ids) > 1:
            best_r, best_i = None, -1
            for i in range(len(ids) - 1):
                hit2 = self.rank.get((ids[i], ids[i + 1]))
                if hit2 is not None and (best_r is None or hit2[1] < best_r):
                    best_r, best_i = hit2[1], i
            if best_r is None:
                break
            a, b = ids[best_i], ids[best_i + 1]
            new = self.rank[(a, b)][0]
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new)
                    trace.append(best_r)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        trace.sort()
        res = (ids, trace, unk)
        if len(self._cache) < 500_000:
            self._cache[word] = res
        return res

    def encode(self, text: str) -> list[int]:
        if self.lowercase:
            text = text.lower()
        out: list[int] = []
        for w in pretokenize(text):
            out.extend(self.encode_word(w)[0])
        return out

    def segment(self, text: str) -> list[dict[str, Any]]:
        if self.lowercase:
            text = text.lower()
        out = []
        for w in pretokenize(text):
            for i in self.encode_word(w)[0]:
                v = self.vocab[i]
                out.append({"id": i, "text": v["display"], "kind": v["kind"]})
        return out

    def decode(self, ids: list[int]) -> str:
        if self.mode == "byte":
            buf = bytearray()
            for i in ids:
                v = self.vocab[i]
                if i == UNK:
                    buf.extend("\ufffd".encode())
                else:
                    buf.extend(bytes(v["raw"]))
            return buf.decode("utf-8", errors="replace")
        return "".join("\ufffd" if i == UNK else self.vocab[i]["raw"] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def _display(raw: Any, mode: str) -> tuple[str, str]:
    """Human readable form + kind for a symbol."""
    if mode == "byte":
        b = bytes(raw)
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(f"<{x:02X}>" for x in b), "bytes"
        if len(b) == 1 and (b[0] < 32 or b[0] == 127):
            return f"<{b[0]:02X}>", "control"
    else:
        s = raw
    if s.isspace():
        return s.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥"), "space"
    if s.strip().isalpha():
        return s.replace(" ", "␣"), "word" if s.startswith(" ") else "piece"
    if s.strip().isdigit():
        return s.replace(" ", "␣"), "number"
    return s.replace(" ", "␣").replace("\n", "⏎"), "mixed"


def train(word_freq: Counter, mode: str = "byte", vocab_size: int = 2000, min_freq: int = 2, lowercase: bool = False, progress: Optional[Callable[[float, str, int, int], None]] = None, checkpoint_every: Optional[int] = None) -> BPEModel:
    if mode not in ("byte", "char"):
        raise ValueError("mode must be byte or char")
    # base alphabet
    vocab: list[dict[str, Any]] = [{"id": UNK, "raw": None, "display": "<unk>", "kind": "special", "freq": 0}]
    if mode == "byte":
        for b in range(256):
            disp, kind = _display([b], mode)
            vocab.append({"id": b + 1, "raw": [b], "display": disp, "kind": kind, "freq": 0})
        unit_of: dict[Any, int] = {}
    else:
        char_freq: Counter = Counter()
        for w, f in word_freq.items():
            for ch in w:
                char_freq[ch] += f
        unit_of = {}
        for ch, _ in char_freq.most_common():
            disp, kind = _display(ch, mode)
            unit_of[ch] = len(vocab)
            vocab.append({"id": len(vocab), "raw": ch, "display": disp, "kind": kind, "freq": 0})
    base_size = len(vocab)

    words: list[list[int]] = []
    freqs: list[int] = []
    for w, f in word_freq.items():
        ids = [b + 1 for b in w.encode("utf-8")] if mode == "byte" else [unit_of[ch] for ch in w]
        words.append(ids)
        freqs.append(f)
        for i in ids:
            vocab[i]["freq"] += f
    total_tokens = sum(len(w) * f for w, f in zip(words, freqs))
    start_tokens = total_tokens

    pair_count: dict[tuple[int, int], int] = defaultdict(int)
    pair_words: dict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_count[p] += f
            pair_words[p].add(wi)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int, int, int]] = []
    checkpoints: list[dict[str, Any]] = [{"vocab": base_size, "tokens": total_tokens, "merge_freq": None}]
    target = max(vocab_size - base_size, 0)
    every = checkpoint_every or max(5, target // 80)
    stop = "vocab_size reached"
    while len(vocab) < vocab_size:
        pair = None
        while heap:
            negc, p = heapq.heappop(heap)
            c = pair_count.get(p, 0)
            if c <= 0:
                continue
            if c == -negc:
                pair = p
                break
            heapq.heappush(heap, (-c, p))  # stale entry: re-insert with the current count
        if pair is None:
            stop = "no pairs left"
            break
        c = pair_count[pair]
        if c < min_freq:
            stop = f"next pair below min_freq ({c} < {min_freq})"
            break
        a, b = pair
        new_id = len(vocab)
        raw = (vocab[a]["raw"] + vocab[b]["raw"]) if mode == "byte" else (vocab[a]["raw"] + vocab[b]["raw"])
        disp, kind = _display(raw, mode)
        vocab.append({"id": new_id, "raw": raw, "display": disp, "kind": kind, "freq": c})
        merges.append((a, b, new_id, c))
        vocab[a]["freq"] -= c
        vocab[b]["freq"] -= c
        for wi in list(pair_words[pair]):
            w = words[wi]
            f = freqs[wi]
            # subtract the pairs of the old word
            for i in range(len(w) - 1):
                q = (w[i], w[i + 1])
                pair_count[q] -= f
            # rewrite
            nw: list[int] = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(new_id)
                    i += 2
                else:
                    nw.append(w[i])
                    i += 1
            words[wi] = nw
            for i in range(len(nw) - 1):
                q = (nw[i], nw[i + 1])
                pair_count[q] += f
                pair_words[q].add(wi)
                heapq.heappush(heap, (-pair_count[q], q))
        total_tokens -= c
        del pair_count[pair]
        del pair_words[pair]
        n = len(merges)
        if n % every == 0 or len(vocab) == vocab_size:
            checkpoints.append({"vocab": len(vocab), "tokens": total_tokens, "merge_freq": c})
        if progress and (n % max(1, target // 50) == 0 or len(vocab) == vocab_size):
            progress(100.0 * n / max(1, target), f"merge {n}/{target} · freq {c}", len(vocab), total_tokens)
    if checkpoints[-1]["vocab"] != len(vocab):
        checkpoints.append({"vocab": len(vocab), "tokens": total_tokens, "merge_freq": merges[-1][3] if merges else None})
    stats = {"stop_reason": stop, "train_words": sum(freqs), "distinct_words": len(words), "start_tokens": start_tokens, "final_tokens": total_tokens}
    return BPEModel(mode, lowercase, base_size, vocab, merges, checkpoints, stats)


def evaluate(model: BPEModel, texts: Iterable[str], curve_vocabs: Optional[list[int]] = None, top_k: int = 2000) -> dict[str, Any]:
    """Coverage / compression metrics, plus the compression curve for smaller vocabularies via merge-rank traces."""
    curve_vocabs = sorted(set(curve_vocabs or [c["vocab"] for c in model.checkpoints]))
    chars = bytes_ = tokens = words_n = unk = whole = letter_words = 0
    tokens_at = [0] * len(curve_vocabs)
    tpw_hist: Counter = Counter()
    tok_len: Counter = Counter()
    tok_freq: Counter = Counter()
    for t in texts:
        if model.lowercase:
            t = t.lower()
        chars += len(t)
        bytes_ += len(t.encode("utf-8"))
        for w in pretokenize(t):
            ids, trace, u = model.encode_word(w)
            n = len(ids)
            tokens += n
            unk += u
            tok_freq.update(ids)
            for i in ids:
                tok_len[min(len(model.vocab[i]["display"]), 10)] += 1
            if w.strip().isalpha():
                letter_words += 1
                tpw_hist[min(n, 8)] += 1
                if n == 1:
                    whole += 1
            words_n += 1
            L0 = len(model._units(w))
            # tokens at vocabulary k = L0 - #(applied merges with rank < k - base_size)
            j = 0
            for ci, k in enumerate(curve_vocabs):
                limit = k - model.base_size
                while j < len(trace) and trace[j] < limit:
                    j += 1
                tokens_at[ci] += L0 - j
    curve = [{"vocab": k, "tokens": tokens_at[i], "chars_per_token": chars / max(1, tokens_at[i]), "tokens_per_word": tokens_at[i] / max(1, words_n)} for i, k in enumerate(curve_vocabs)]
    # cumulative coverage by most frequent tokens
    cum, acc = [], 0
    for rank, (_tid, f) in enumerate(tok_freq.most_common(top_k), start=1):
        acc += f
        if rank <= 50 or rank % max(1, top_k // 200) == 0 or rank == len(tok_freq):
            cum.append({"k": rank, "coverage": acc / max(1, tokens)})
    return {
        "chars": chars,
        "bytes": bytes_,
        "tokens": tokens,
        "words": words_n,
        "unk_units": unk,
        "char_coverage": 1.0 - (unk / max(1, chars if model.mode == "char" else bytes_)),
        "chars_per_token": chars / max(1, tokens),
        "bytes_per_token": bytes_ / max(1, tokens),
        "tokens_per_word": tokens / max(1, words_n),
        "whole_word_rate": whole / max(1, letter_words),
        "vocab_used": len(tok_freq),
        "vocab_size": model.vocab_size,
        "curve": curve,
        "top_k_coverage": cum,
        "tokens_per_word_hist": [{"bin": f"{i}" if i < 8 else "8+", "count": tpw_hist.get(i, 0)} for i in range(1, 9)],
        "token_length_hist": [{"bin": f"{i}" if i < 10 else "10+", "count": tok_len.get(i, 0)} for i in range(1, 11)],
        "top_tokens": [{"id": tid, "text": model.vocab[tid]["display"], "kind": model.vocab[tid]["kind"], "count": f} for tid, f in tok_freq.most_common(60)],
    }
