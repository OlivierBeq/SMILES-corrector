from uncorrupt_smiles.utils.tokenizer import smi_tokenizer
from uncorrupt_smiles.vocab import EOS, PAD, SOS, UNK, Vocab

LINES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C1CCCCC1"]


def test_specials_always_present():
    v = Vocab.build_from_lines(LINES, max_size=5)
    for special in (PAD, SOS, EOS, UNK):
        assert special in v.itos


def test_max_size_is_respected():
    v = Vocab.build_from_lines(LINES, max_size=6)
    assert len(v) <= 6


def test_encode_decode_round_trip():
    v = Vocab.build_from_lines(LINES, max_size=50)
    for smi in LINES:
        tokens = smi_tokenizer(smi)
        ids = v.encode(tokens)
        assert ids[0] == v.sos_idx
        assert ids[-1] == v.eos_idx
        decoded = v.decode(ids)
        assert "".join(decoded) == smi


def test_decode_stops_at_eos_and_skips_pad_and_sos():
    v = Vocab.build_from_lines(LINES, max_size=50)
    ids = [v.sos_idx, v.stoi["C"], v.stoi["C"], v.eos_idx, v.pad_idx, v.pad_idx]
    decoded = v.decode(ids)
    assert decoded == ["C", "C"]


def test_unknown_token_maps_to_unk():
    v = Vocab.build_from_lines(["CCO"], max_size=50)
    ids = v.encode(["Z"])  # "Z" never appears in training lines
    assert v.unk_idx in ids


def test_save_load_round_trip(tmp_path):
    v = Vocab.build_from_lines(LINES, max_size=50)
    path = str(tmp_path / "vocab.pt")
    v.save(path)
    v2 = Vocab.load(path)
    assert v2.itos == v.itos
