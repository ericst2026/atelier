"""Starter: a character-level tokenizer. Correct, tiny, and slow to compress.

Replace `train` / `encode` / `decode` with your own algorithm. Keep the interface."""
from collections import Counter


class Tokenizer:
    def __init__(self) -> None:
        self.stoi: dict[str, int] = {"<unk>": 0}
        self.itos: list[str] = ["<unk>"]

    def train(self, texts: list[str], vocab_size: int) -> None:
        counts = Counter(ch for t in texts for ch in t)
        for ch, _ in counts.most_common(vocab_size - 1):
            self.stoi[ch] = len(self.itos)
            self.itos.append(ch)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] if 0 < i < len(self.itos) else "\ufffd" for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
