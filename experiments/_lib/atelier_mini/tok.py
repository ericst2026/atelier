"""The tokenizer students trained in the Tokenizer experiment, wrapped for the model.

Ids 0–3 are reserved control tokens, so a BPE of size V becomes a vocabulary of
V+4. Everything downstream (packing, SFT masks, generation) speaks these ids."""
from pathlib import Path
from typing import Iterable

from atelier_nlp import bpe

PAD, BOS, EOS, SEP = 0, 1, 2, 3
SPECIAL = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<sep>": SEP}
OFFSET = 4


class MiniTokenizer:
    def __init__(self, model: bpe.BPEModel):
        self.bpe = model
        self.vocab_size = model.vocab_size + OFFSET

    @classmethod
    def load(cls, path: str | Path) -> "MiniTokenizer":
        return cls(bpe.BPEModel.load(path))

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [i + OFFSET for i in self.bpe.encode(text)]
        if bos:
            ids = [BOS] + ids
        if eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.bpe.decode([i - OFFSET for i in ids if i >= OFFSET])

    def pieces(self, text: str) -> list[dict]:
        """For the token view in the UI."""
        return self.bpe.segment(text)


class HFTokenizer:
    """A tokenizer.json from the materials folder (gpt2, SmolLM2, …), wrapped in the
    same interface as MiniTokenizer so the model code does not care where it came from."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as _T

        self.tk = _T.from_file(str(path))
        self.base = self.tk.get_vocab_size()
        self.vocab_size = self.base + OFFSET

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [i + OFFSET for i in self.tk.encode(text).ids]
        if bos:
            ids = [BOS] + ids
        if eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tk.decode([i - OFFSET for i in ids if i >= OFFSET])

    def pieces(self, text: str) -> list[dict]:
        enc = self.tk.encode(text)
        return [{"id": i + OFFSET, "text": t.replace("Ġ", "␣").replace("Ċ", "⏎"), "kind": "piece"} for i, t in zip(enc.ids, enc.tokens)]


def load_tokenizer(path: str | Path):
    """MiniTokenizer for a BPE trained in the Tokenizer experiment, HFTokenizer for a
    downloaded tokenizer.json."""
    p = Path(path)
    if p.is_dir():
        p = p / "tokenizer.json"
    import json

    head = json.loads(p.read_text(encoding="utf-8"))
    return MiniTokenizer.load(p) if head.get("format") == "atelier-bpe-1" else HFTokenizer(p)
