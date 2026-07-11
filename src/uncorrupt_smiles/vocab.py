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
    """Bidirectional mapping between SMILES tokens and integer ids.

    :param itos: Index-to-string table; indices 0-3 are expected to be the
        special tokens :data:`PAD`, :data:`SOS`, :data:`EOS`, :data:`UNK` in
        that order, as produced by :meth:`build_from_lines`.
    """

    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        """
        :return: Number of tokens in the vocabulary.
        """
        return len(self.itos)

    def __eq__(self, other) -> bool:
        """
        :param other: Object to compare against.
        :return: ``True`` if `other` is a :class:`Vocab` with the same
            index-to-string table.
        """
        return isinstance(other, Vocab) and self.itos == other.itos

    @property
    def pad_idx(self) -> int:
        """
        :return: Index of the ``<pad>`` special token.
        """
        return self.stoi[PAD]

    @property
    def sos_idx(self) -> int:
        """
        :return: Index of the ``<sos>`` special token.
        """
        return self.stoi[SOS]

    @property
    def eos_idx(self) -> int:
        """
        :return: Index of the ``<eos>`` special token.
        """
        return self.stoi[EOS]

    @property
    def unk_idx(self) -> int:
        """
        :return: Index of the ``<unk>`` special token.
        """
        return self.stoi[UNK]

    def encode(self, tokens: list[str]) -> list[int]:
        """Wraps tokens with ``<sos>``/``<eos>``, mapping unknown tokens to ``<unk>``.

        :param tokens: SMILES tokens to encode, e.g. as produced by
            :func:`~uncorrupt_smiles.utils.tokenizer.smi_tokenizer`.
        :return: Token ids, prefixed with :attr:`sos_idx` and suffixed with
            :attr:`eos_idx`.
        """
        unk = self.unk_idx
        ids = [self.stoi.get(tok, unk) for tok in tokens]
        return [self.sos_idx, *ids, self.eos_idx]

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> list[str]:
        """Maps ids back to tokens, dropping a leading ``<sos>`` and (optionally)
        truncating at the first ``<eos>``.

        :param ids: Token ids to decode.
        :param stop_at_eos: If ``True``, stop decoding at the first ``<eos>``
            token instead of including it and whatever follows.
        :return: Decoded tokens, with ``<sos>`` and ``<pad>`` removed.
        """
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
        """Encodes tokens and wraps the result in a tensor.

        :param tokens: SMILES tokens to encode via :meth:`encode`.
        :param device: Device to allocate the tensor on; ``None`` uses the
            current default device.
        :return: 1-D tensor of token ids, dtype ``torch.long``.
        """
        return torch.tensor(self.encode(tokens), dtype=torch.long, device=device)

    @classmethod
    def build_from_lines(
        cls,
        lines: Iterable[str],
        tokenizer=smi_tokenizer,
        max_size: int = 200,
        min_freq: int = 1,
    ) -> "Vocab":
        """Builds a vocabulary by counting token frequencies over a stream of lines.

        :param lines: Raw SMILES strings to tokenize and count. Consumed as a
            stream, so memory use is bounded by the resulting vocabulary size
            rather than the number of lines.
        :param tokenizer: Callable turning a SMILES string into a list of
            tokens; defaults to :func:`~uncorrupt_smiles.utils.tokenizer.smi_tokenizer`.
        :param max_size: Maximum vocabulary size, including the special
            tokens.
        :param min_freq: Minimum token frequency required for a token to be
            included.
        :return: A new vocabulary whose index-to-string table starts with
            :data:`SPECIALS` followed by the most frequent qualifying tokens.
        """
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
        """Persists the index-to-string table to disk.

        :param path: Destination file path to write to, readable back via
            :meth:`load`.
        """
        torch.save(self.itos, path)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Loads a vocabulary previously written by :meth:`save`.

        :param path: Path to a vocabulary file previously written by
            :meth:`save`.
        :return: The reconstructed vocabulary.
        """
        return cls(torch.load(path, weights_only=True))
