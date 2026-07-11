import csv

from rdkit import Chem

from uncorrupt_smiles.preprocess import standardize_smiles, standardize_stream


def test_standardize_smiles_valid_input_is_canonical():
    out = standardize_smiles("C(C)O")  # non-canonical ethanol
    assert out is not None
    assert Chem.MolToSmiles(Chem.MolFromSmiles("CCO")) == out


def test_standardize_smiles_rejects_garbage():
    assert standardize_smiles("not a smiles!!") is None


def test_standardize_smiles_rejects_empty_and_none():
    assert standardize_smiles("") is None
    assert standardize_smiles(None) is None
    assert standardize_smiles(123) is None


def test_standardize_stream_drops_and_dedupes(tmp_path):
    rows = ["CCO", "OCC", "garbage!!", "", "c1ccccc1", "C" * 300]
    input_csv = tmp_path / "in.csv"
    with open(input_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SMILES"])
        for r in rows:
            w.writerow([r])

    output_csv = tmp_path / "out.csv"
    standardize_stream(str(input_csv), "SMILES", str(output_csv), length_threshold=50)

    with open(output_csv) as f:
        out_rows = [row["STD_SMILES"] for row in csv.DictReader(f)]

    # CCO and OCC canonicalize to the same molecule -> only one survives
    assert out_rows.count(Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))) == 1
    assert Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1")) in out_rows
    # garbage/empty/too-long all dropped
    assert len(out_rows) == 2
    for smi in out_rows:
        assert Chem.MolFromSmiles(smi) is not None


def test_standardize_stream_respects_separator(tmp_path):
    input_csv = tmp_path / "semi.csv"
    with open(input_csv, "w", newline="") as f:
        f.write("SMILES;OTHER\nCCO;x\nc1ccccc1;y\n")
    output_csv = tmp_path / "out.csv"
    standardize_stream(str(input_csv), "SMILES", str(output_csv), separator=";")
    with open(output_csv) as f:
        out_rows = [row["STD_SMILES"] for row in csv.DictReader(f)]
    assert len(out_rows) == 2
