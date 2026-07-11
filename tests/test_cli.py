import csv
import os

from uncorrupt_smiles.cli import main


def test_standardize_subcommand(smiles_csv, tmp_path):
    output_csv = str(tmp_path / "std_out.csv")
    main(["standardize", "--input-csv", smiles_csv, "--smiles-col", "SMILES",
          "--output-csv", output_csv, "--threshold", "100"])
    assert os.path.exists(output_csv)
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0


def test_generate_errors_subcommand(std_csv, fragment_csv, tmp_path):
    train_csv = str(tmp_path / "train.csv")
    dev_csv = str(tmp_path / "dev.csv")
    main(["generate-errors", "--input-csv", std_csv, "--smiles-col", "STD_SMILES",
          "--fragment-csv", fragment_csv, "--fragment-col", "FRAGMENT",
          "--fragment-pool-size", "5", "--train-csv", train_csv, "--dev-csv", dev_csv,
          "--frac-train", "0.8"])
    assert os.path.exists(train_csv)
    assert os.path.exists(dev_csv)
    with open(train_csv) as f:
        assert len(list(csv.DictReader(f))) > 0


def test_train_and_fix_subcommands(errors_csv_pair, smiles_csv, tmp_path):
    train_csv, dev_csv = errors_csv_pair
    ckpt = str(tmp_path / "ckpt.pkg")
    main(["train", "--train-csv", train_csv, "--dev-csv", dev_csv,
          "--hid-dim", "16", "--n-layers", "1", "--n-heads", "2", "--pf-dim", "32",
          "--batch-size", "4", "--epochs", "1", "--device", "cpu",
          "--checkpoint-out", ckpt, "--shuffle-buffer", "8", "--max-length", "64"])
    assert os.path.exists(ckpt)

    fixed_csv = str(tmp_path / "fixed.csv")
    main(["fix", "--checkpoint", ckpt, "--input-csv", smiles_csv, "--smiles-col", "SMILES",
          "--output-csv", fixed_csv, "--device", "cpu", "--batch-size", "8"])
    assert os.path.exists(fixed_csv)
    with open(smiles_csv) as f:
        n_input = sum(1 for _ in csv.DictReader(f))
    with open(fixed_csv) as f:
        assert len(list(csv.DictReader(f))) == n_input


def test_train_resume_subcommand(errors_csv_pair, tmp_path):
    train_csv, dev_csv = errors_csv_pair
    ckpt1 = str(tmp_path / "ckpt1.pkg")
    main(["train", "--train-csv", train_csv, "--dev-csv", dev_csv,
          "--hid-dim", "16", "--n-layers", "1", "--n-heads", "2", "--pf-dim", "32",
          "--batch-size", "4", "--epochs", "1", "--device", "cpu",
          "--checkpoint-out", ckpt1, "--max-length", "64"])

    ckpt2 = str(tmp_path / "ckpt2.pkg")
    main(["train", "--train-csv", train_csv, "--dev-csv", dev_csv, "--resume", ckpt1,
          "--batch-size", "4", "--epochs", "1", "--device", "cpu", "--checkpoint-out", ckpt2])
    assert os.path.exists(ckpt2)
