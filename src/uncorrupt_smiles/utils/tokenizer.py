import re

_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def smi_tokenizer(smi: str, reverse: bool = False) -> list[str]:
    """Tokenize a SMILES string into its atom/bond/ring-closure symbols."""
    tokens = _PATTERN.findall(smi)
    assert smi == "".join(tokens), f"tokenization did not round-trip for {smi!r}"
    if reverse:
        return tokens[::-1]
    return tokens
