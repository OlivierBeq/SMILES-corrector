import csv

import pytest

VALID_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "C1CCCCC1", "CCN(CC)CC",
    "c1ccc2ccccc2c1", "Cc1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCN", "CCCC",
    "c1ccncc1", "CC(C)O", "CCOCC", "CC#N", "c1ccsc1", "OCC(O)CO", "CC(C)(C)O",
    "c1ccoc1", "CCCl", "CCBr", "CC(=O)N", "c1cc[nH]c1", "CCS", "CCF", "CC(C)C",
]

FRAGMENTS = ["CC", "CCC", "c1ccccc1", "CO", "N", "CCO", "c1ccncc1", "CCN"]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header])
        for row in rows:
            writer.writerow([row])
    return str(path)


@pytest.fixture
def smiles_csv(tmp_path):
    """A single-column SMILES CSV (header 'SMILES'), for standardize/fix-style inputs."""
    return _write_csv(tmp_path / "smiles.csv", "SMILES", VALID_SMILES)


@pytest.fixture
def std_csv(tmp_path):
    """A single-column standardized-SMILES CSV (header 'STD_SMILES')."""
    return _write_csv(tmp_path / "std.csv", "STD_SMILES", VALID_SMILES)


@pytest.fixture
def fragment_csv(tmp_path):
    return _write_csv(tmp_path / "fragments.csv", "FRAGMENT", FRAGMENTS)


@pytest.fixture
def errors_csv_pair(tmp_path, std_csv, fragment_csv):
    from uncorrupt_smiles.invalid_smiles import write_errors_split

    train_csv = str(tmp_path / "err_train.csv")
    dev_csv = str(tmp_path / "err_dev.csv")
    write_errors_split(
        std_csv, "STD_SMILES", fragment_csv, "FRAGMENT", train_csv, dev_csv,
        seed=42, fragment_pool_size=10, frac_train=0.8,
    )
    return train_csv, dev_csv
