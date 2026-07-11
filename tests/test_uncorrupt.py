from pathlib import Path
from unittest.mock import patch

from uncorrupt_smiles.data import iter_csv_column
from uncorrupt_smiles.transformer import Seq2Seq
from uncorrupt_smiles.uncorrupt import Uncorrupt
from uncorrupt_smiles.vocab import Vocab

TINY_KWARGS = dict(hid_dim=16, n_layers=1, n_heads=2, pf_dim=32, dropout=0.1)


def _make_checkpoint(train_csv, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    src_vocab = Vocab.build_from_lines(iter_csv_column(train_csv, "ERROR"), max_size=100)
    trg_vocab = Vocab.build_from_lines(iter_csv_column(train_csv, "STD_SMILES"), max_size=100)
    model = Seq2Seq.build(
        len(src_vocab), len(trg_vocab), max_length=64, device="cpu",
        src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx, **TINY_KWARGS,
    )
    model.src_vocab, model.trg_vocab = src_vocab, trg_vocab
    model.save_checkpoint(path)


def test_uncorrupt_loads_explicit_checkpoint_and_fixes_smiles(errors_csv_pair, tmp_path):
    train_csv, _ = errors_csv_pair
    ckpt_path = str(tmp_path / "ckpt.pkg")
    _make_checkpoint(train_csv, ckpt_path)

    uncorrupt = Uncorrupt(checkpoint=ckpt_path, device="cpu")
    single = uncorrupt.fix_smiles("CCO", max_len=10)
    multi = uncorrupt.fix_smiles(["CCO", "c1ccccc1"], max_len=10)
    assert isinstance(single, list) and len(single) == 1
    assert isinstance(multi, list) and len(multi) == 2


def test_uncorrupt_fix_smiles_csv(errors_csv_pair, smiles_csv, tmp_path):
    train_csv, _ = errors_csv_pair
    ckpt_path = str(tmp_path / "ckpt.pkg")
    _make_checkpoint(train_csv, ckpt_path)

    output_csv = str(tmp_path / "fixed.csv")
    uncorrupt = Uncorrupt(checkpoint=ckpt_path, device="cpu")
    uncorrupt.fix_smiles_csv(smiles_csv, "SMILES", output_csv, batch_size=4)

    import csv as csv_mod
    with open(smiles_csv) as f:
        n_input = sum(1 for _ in csv_mod.DictReader(f))
    with open(output_csv) as f:
        assert len(list(csv_mod.DictReader(f))) == n_input


def test_uncorrupt_downloads_default_checkpoint_when_missing(errors_csv_pair, tmp_path, monkeypatch):
    train_csv, _ = errors_csv_pair
    default_dest = tmp_path / "data" / "performance" / "default.pkg"

    def _fake_fetch_all(only=None, **kwargs):
        _make_checkpoint(train_csv, str(default_dest))

    monkeypatch.chdir(tmp_path)
    with patch("uncorrupt_smiles.uncorrupt.DEFAULT_CHECKPOINT", "data/performance/default.pkg"), \
         patch("uncorrupt_smiles.uncorrupt.fetch_all", side_effect=_fake_fetch_all) as mock_fetch:
        uncorrupt = Uncorrupt(device="cpu")
        mock_fetch.assert_called_once()
    assert uncorrupt.checkpoint == "data/performance/default.pkg"
