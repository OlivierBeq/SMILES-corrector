import csv
import random

import pytest
from rdkit import Chem

from uncorrupt_smiles.invalidSMILES import (
    INVALID_TYPES,
    build_seed_vocab,
    generate_errors,
    introduce_error,
    reservoir_sample_fragments,
    write_errors_split,
)
from uncorrupt_smiles.utils.tokenizer import smi_tokenizer

MOLECULES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C1CCCCC1", "CCN(CC)CC",
    "c1ccc2ccccc2c1", "Cc1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
]


@pytest.mark.parametrize("smi", MOLECULES)
def test_introduce_error_eventually_produces_invalid_smiles(smi):
    random.seed(0)
    vocab = set(smi_tokenizer(smi))
    invalid, target = introduce_error(smi, "CC", vocab, invalid_type="all", num_errors=1)
    assert invalid is None or Chem.MolFromSmiles(invalid) is None
    assert Chem.MolFromSmiles(target) is not None


def test_introduce_error_multiple_applies_extra_corruptions():
    random.seed(1)
    vocab = set(smi_tokenizer("CC(=O)Oc1ccccc1C(=O)O"))
    invalid, target = introduce_error(
        "CC(=O)Oc1ccccc1C(=O)O", "CC", vocab, invalid_type="multiple", num_errors=3
    )
    # multiple should still end up invalid (or fail entirely -> None), never a no-op
    if invalid is not None:
        assert Chem.MolFromSmiles(invalid) is None


def test_reservoir_sample_respects_k():
    values = [f"frag{i}" for i in range(100)]
    import tempfile
    import os
    path = tempfile.mktemp(suffix=".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FRAGMENT"])
        for v in values:
            w.writerow([v])
    pool = reservoir_sample_fragments(path, "FRAGMENT", k=10, seed=1)
    assert len(pool) == 10
    assert all(v in values for v in pool)
    os.remove(path)


def test_reservoir_sample_smaller_than_k_returns_all(fragment_csv):
    pool = reservoir_sample_fragments(fragment_csv, "FRAGMENT", k=1000, seed=1)
    assert len(pool) == 8  # len(FRAGMENTS) in conftest


def test_build_seed_vocab_skips_unparseable():
    smiles = iter(["CCO", "garbage!!", "c1ccccc1"])
    vocab = build_seed_vocab(smiles, sample_size=10)
    assert "C" in vocab
    assert "c" in vocab


def test_generate_errors_skips_unparseable_input():
    pairs = list(generate_errors(["CCO", "not a smiles", ""], ["CC"], seed=0))
    assert len(pairs) == 1
    target, invalid = pairs[0]
    assert Chem.MolFromSmiles(target) is not None
    assert Chem.MolFromSmiles(invalid) is None


def test_generate_errors_invalid_type_validation():
    with pytest.raises(ValueError):
        list(generate_errors(["CCO"], ["CC"], seed=0, invalid_type="not-a-real-type"))


@pytest.mark.parametrize("invalid_type", INVALID_TYPES)
def test_generate_errors_runs_for_every_invalid_type(invalid_type):
    pairs = list(generate_errors(MOLECULES, ["CC", "CCC"], seed=0, invalid_type=invalid_type))
    for target, invalid in pairs:
        assert Chem.MolFromSmiles(target) is not None
        assert Chem.MolFromSmiles(invalid) is None


def test_write_errors_split_header_and_determinism(std_csv, fragment_csv, tmp_path):
    train_a = str(tmp_path / "a_train.csv")
    dev_a = str(tmp_path / "a_dev.csv")
    write_errors_split(
        std_csv, "STD_SMILES", fragment_csv, "FRAGMENT", train_a, dev_a,
        seed=7, fragment_pool_size=10, frac_train=0.8,
    )
    with open(train_a) as f:
        rows_a = list(csv.reader(f))
    assert rows_a[0] == ["STD_SMILES", "ERROR"]
    assert len(rows_a) > 1

    train_b = str(tmp_path / "b_train.csv")
    dev_b = str(tmp_path / "b_dev.csv")
    write_errors_split(
        std_csv, "STD_SMILES", fragment_csv, "FRAGMENT", train_b, dev_b,
        seed=7, fragment_pool_size=10, frac_train=0.8,
    )
    with open(train_b) as f:
        rows_b = list(csv.reader(f))
    assert rows_a == rows_b  # same seed -> identical output


def test_write_errors_split_rows_are_valid_target_invalid_error(errors_csv_pair):
    train_csv, dev_csv = errors_csv_pair
    for path in (train_csv, dev_csv):
        with open(path) as f:
            for row in csv.DictReader(f):
                assert Chem.MolFromSmiles(row["STD_SMILES"]) is not None
                assert Chem.MolFromSmiles(row["ERROR"]) is None
