from __future__ import annotations

import re

_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def smi_tokenizer(smi: str, reverse: bool = False) -> list[str]:
    """Tokenize a SMILES string into its atom/bond/ring-closure symbols.

    :param smi: SMILES string to tokenize.
    :param reverse: If ``True``, return the tokens in reverse order; used
        when tokenizing target SMILES for training/decoding (see
        :meth:`~uncorrupt_smiles.data.SmilesPairIterableDataset._encode`).
    :return: The SMILES tokens, in forward or reverse order.
    :raises AssertionError: If the concatenated tokens do not round-trip to
        `smi` exactly, meaning some character in `smi` was not matched by
        the tokenizer's pattern.
    """
    tokens = _PATTERN.findall(smi)
    assert smi == "".join(tokens), f"tokenization did not round-trip for {smi!r}"
    if reverse:
        return tokens[::-1]
    return tokens
