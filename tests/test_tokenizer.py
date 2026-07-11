import pytest

from uncorrupt_smiles.utils.tokenizer import smi_tokenizer

SMILES_EXAMPLES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCCCC1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "c1cc[nH]c1",
    "CC#N",
    "[nH]1cccc1",
    "CC%10CC%10",
]


@pytest.mark.parametrize("smi", SMILES_EXAMPLES)
def test_round_trips(smi):
    tokens = smi_tokenizer(smi)
    assert "".join(tokens) == smi


def test_bracket_atom_is_one_token():
    tokens = smi_tokenizer("[nH]1cccc1")
    assert tokens[0] == "[nH]"


def test_percent_ring_closure_is_one_token():
    tokens = smi_tokenizer("CC%10CC%10")
    assert "%10" in tokens


def test_reverse_is_exact_reverse_of_forward():
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    forward = smi_tokenizer(smi)
    backward = smi_tokenizer(smi, reverse=True)
    assert backward == forward[::-1]


def test_invalid_characters_raise():
    with pytest.raises(AssertionError):
        smi_tokenizer("not a smiles!!")
