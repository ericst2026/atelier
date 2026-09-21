"""Exact + near-duplicate detection: normalized hashing, then MinHash signatures
bucketed with locality-sensitive hashing, candidates verified with true Jaccard,
and union-find to turn pairs into clusters."""
import hashlib
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np

from .text import normalize

NUM_HASHES = 128
MERSENNE = (1 << 61) - 1
LSH_CONFIGS = [(64, 2), (32, 4), (16, 8), (8, 16)]  # (bands, rows)


def shingles(text: str, kind: str = "word", n: int = 3) -> set[str]:
    if kind == "char":
        t = text
        return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}
    toks = text.split()
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _h32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


class MinHasher:
    def __init__(self, num_hashes: int = NUM_HASHES, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, (1 << 32) - 1, size=num_hashes, dtype=np.uint64)
        self.b = rng.integers(0, (1 << 32) - 1, size=num_hashes, dtype=np.uint64)
        self.n = num_hashes

    def signature(self, items: set[str]) -> np.ndarray:
        if not items:
            return np.full(self.n, MERSENNE, dtype=np.uint64)
        x = np.fromiter((_h32(s) for s in items), dtype=np.uint64, count=len(items))
        # (a*x + b) mod p with a,x,b < 2^32 keeps every intermediate below 2^64
        hv = (self.a[:, None] * x[None, :] + self.b[:, None]) % np.uint64(MERSENNE)
        return hv.min(axis=1)


def pick_lsh(threshold: float) -> tuple[int, int]:
    target = threshold * 0.85
    best = LSH_CONFIGS[0]
    for b, r in LSH_CONFIGS:
        if (1.0 / b) ** (1.0 / r) <= target:
            best = (b, r)
    return best


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dedupe(texts: list[str], lowercase: bool = True, collapse_ws: bool = True, strip_punct: bool = False, near: bool = True, shingle: str = "word", n: int = 3, threshold: float = 0.7, keep: str = "first", max_bucket: int = 400, progress=None, on_pairs=None) -> dict[str, Any]:
    """on_pairs(checked, total, similar), when given, is called as the near pass
    checks the candidate pairs LSH found: how many it has checked, of how many,
    and how many of those turned out similar enough to be duplicates."""
    N = len(texts)
    norm = [normalize(t, lowercase, collapse_ws, strip_punct) for t in texts]
    # pass 1: exact duplicates after normalization
    seen: dict[str, int] = {}
    removed_exact: dict[int, int] = {}
    survivors: list[int] = []
    for i, t in enumerate(norm):
        key = hashlib.blake2b(t.encode("utf-8"), digest_size=16).hexdigest()
        if key in seen:
            removed_exact[i] = seen[key]
        else:
            seen[key] = i
            survivors.append(i)
    if progress:
        progress(20, f"exact pass: {len(removed_exact)} duplicates")
    removed_near: dict[int, tuple[int, float]] = {}
    clusters: list[list[int]] = []
    sim_values: list[float] = []
    candidates = 0
    lsh = None
    if near and len(survivors) > 1:
        sh = {i: shingles(norm[i], shingle, n) for i in survivors}
        hasher = MinHasher()
        sigs = {i: hasher.signature(sh[i]) for i in survivors}
        if progress:
            progress(45, f"minhash signatures for {len(survivors)} documents")
        bands, rows = pick_lsh(threshold)
        lsh = {"bands": bands, "rows": rows, "lsh_threshold": round((1.0 / bands) ** (1.0 / rows), 3)}
        buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
        for i in survivors:
            s = sigs[i]
            for b in range(bands):
                buckets[(b, s[b * rows : (b + 1) * rows].tobytes())].append(i)
        pairs: set[tuple[int, int]] = set()
        for members in buckets.values():
            if len(members) < 2:
                continue
            members = members[:max_bucket]
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pairs.add((members[x], members[y]))
        candidates = len(pairs)
        if progress:
            progress(65, f"{candidates} candidate pairs from LSH")
        uf = UnionFind(N)
        pair_sim: dict[tuple[int, int], float] = {}
        every = max(1, candidates // 40)
        for k, (i, j) in enumerate(pairs, 1):
            s = jaccard(sh[i], sh[j])
            sim_values.append(s)
            if s >= threshold:
                uf.union(i, j)
                pair_sim[(i, j)] = s
            if on_pairs and (k % every == 0 or k == candidates):
                on_pairs(k, candidates, len(pair_sim))
        groups: dict[int, list[int]] = defaultdict(list)
        for i in survivors:
            groups[uf.find(i)].append(i)
        for root, members in groups.items():
            if len(members) < 2:
                continue
            members.sort()
            if keep == "longest":
                keeper = max(members, key=lambda k: (len(texts[k]), -k))
            else:
                keeper = members[0]
            clusters.append(members)
            for m in members:
                if m == keeper:
                    continue
                s = pair_sim.get((min(m, keeper), max(m, keeper)))
                if s is None:  # linked transitively: report best similarity to any kept neighbour
                    s = max((pair_sim.get((min(m, o), max(m, o)), 0.0) for o in members if o != m), default=0.0)
                removed_near[m] = (keeper, s)
    if progress:
        progress(90, f"near pass: {len(removed_near)} duplicates")
    kept = [i for i in survivors if i not in removed_near]
    bins = 20
    hist = [0] * bins
    for s in sim_values:
        hist[min(int(s * bins), bins - 1)] += 1
    sim_hist = [{"bin": f"{i / bins:.2f}", "lo": i / bins, "count": hist[i]} for i in range(bins)]
    cluster_hist = Counter(min(len(c), 10) for c in clusters)
    return {
        "kept": kept,
        "removed_exact": removed_exact,
        "removed_near": removed_near,
        "clusters": clusters,
        "candidates": candidates,
        "sim_hist": sim_hist,
        "cluster_sizes": [{"bin": f"{k}" if k < 10 else "10+", "count": cluster_hist.get(k, 0)} for k in range(2, 11)],
        "lsh": lsh,
        "stats": {
            "docs": N,
            "after_exact": len(survivors),
            "after_near": len(kept),
            "chars": sum(len(t) for t in texts),
            "chars_after_exact": sum(len(texts[i]) for i in survivors),
            "chars_after_near": sum(len(texts[i]) for i in kept),
        },
    }
