"""Character-level vocabulary for SMILES tokens.

Built by streaming a Counter over an iterable of raw SMILES strings, so memory is bounded by
the vocabulary size (a SMILES alphabet is on the order of 100-200 distinct tokens), never by
dataset size.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import torch

from uncorrupt_smiles.utils.tokenizer import smi_tokenizer

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIALS = [PAD, SOS, EOS, UNK]


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __eq__(self, other) -> bool:
        return isinstance(other, Vocab) and self.itos == other.itos

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_idx(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK]

    def encode(self, tokens: list[str]) -> list[int]:
        """Wraps tokens with <sos>/<eos>, mapping unknown tokens to <unk>."""
        unk = self.unk_idx
        ids = [self.stoi.get(tok, unk) for tok in tokens]
        return [self.sos_idx, *ids, self.eos_idx]

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> list[str]:
        """Maps ids back to tokens, dropping a leading <sos> and (optionally) truncating
        at the first <eos>."""
        tokens = []
        for i in ids:
            tok = self.itos[int(i)]
            if tok in (SOS, PAD):
                # <pad> is a structural filler a model can still technically predict
                # (e.g. early in training); it should never appear in decoded output.
                continue
            if tok == EOS and stop_at_eos:
                break
            tokens.append(tok)
        return tokens

    def as_tensor(self, tokens: list[str], device=None) -> torch.Tensor:
        return torch.tensor(self.encode(tokens), dtype=torch.long, device=device)

    @classmethod
    def build_from_lines(
        cls,
        lines: Iterable[str],
        tokenizer=smi_tokenizer,
        max_size: int = 200,
        min_freq: int = 1,
    ) -> "Vocab":
        counts: Counter[str] = Counter()
        for line in lines:
            counts.update(tokenizer(line))
        itos = list(SPECIALS)
        for tok, freq in counts.most_common():
            if freq < min_freq:
                break
            if len(itos) >= max_size:
                break
            itos.append(tok)
        return cls(itos)

    def save(self, path: str) -> None:
        torch.save(self.itos, path)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        return cls(torch.load(path, weights_only=True))
