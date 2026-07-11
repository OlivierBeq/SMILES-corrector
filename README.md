# 🧬 uncorrupt-smiles

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

This project implements the approach described in:

> Schoenmaker, L., Béquignon, O.J.M., Jespers, W. & Gerard J.P. van Westen<br/>
> UnCorrupt SMILES: a novel approach to de novo design.<br/>
> *Journal of Cheminformatics* **15**, 22 (2023). https://doi.org/10.1186/s13321-023-00696-x

**uncorrupt-smiles fixes broken SMILES.** Give it a trained checkpoint and a SMILES string (or a
CSV of them), and it hands back a valid molecule — from the CLI or from Python, one string or a
streamed million-row file.

```python
from uncorrupt_smiles import Uncorrupt

uncorrupt = Uncorrupt()  # downloads and loads the bundled pretrained checkpoint, once
uncorrupt.fix_smiles("C1=CC=CC=C1(")
# -> ["c1ccccc1"]
```

`Uncorrupt` is the main entrypoint: no checkpoint path, vocab, or device to wire up — it
downloads the pretrained checkpoint bundled with this project the first time it's needed (see
[Fetch example data](#-fetch-example-data) below) and picks CUDA automatically if available.
Pass `checkpoint="path/to/your.pkg"` to use a self-trained checkpoint instead. Everything else —
standardizing datasets,
generating synthetic errors, training your own corrector — is secondary tooling for anyone who
wants to build a new model, built around the same three composable, argument-driven,
streaming-first building blocks: no config files, and no dataset is ever loaded fully into
memory ([polars](https://pola.rs) streams every CSV in and out).

## 📦 Installation

Requires Python 3.11+.

```bash
pip install uncorrupt-smiles
```

Optional extras:

```bash
pip install "uncorrupt-smiles[analysis]"   # pandas/matplotlib/seaborn for the analysis/ scripts
pip install "uncorrupt-smiles[test]"       # pytest, for running the test suite
```

To work from a clone instead (e.g. to run the test suite or modify the source):

```bash
git clone https://github.com/OlivierBeq/uncorrupt-smiles.git
cd uncorrupt-smiles
pip install -e ".[test]"
```

Core dependencies — `torch >= 2.4`, `rdkit >= 2024.3.1`, `polars >= 1.0` — are resolved by pip
to whichever wheel (CPU or CUDA) matches your machine.

> **Note:** checkpoints produced by earlier versions of this project (including the archive on
> [Zenodo](https://zenodo.org/record/7157412#.Y1edr3ZBxD8)) use a different, incompatible
> checkpoint format and cannot be loaded by `Seq2Seq.load_checkpoint`. Retrain to obtain a
> checkpoint this version can use.

## 🚀 Usage

Run `uncorrupt-smiles <subcommand> -h` for the full flag list of any subcommand. The CLI and the
Python API expose exactly the same operations, in the same order of importance: fixing SMILES
first, then the supporting tooling for anyone building their own corrector.

### CLI

#### 🩹 Fix

```bash
uncorrupt-smiles fix \
  --checkpoint data/performance/transformer_multiple_12_PAPYRUS_200_16_3.pkg \
  --input-csv generated/gan_ckpt100.csv --smiles-col SMILES \
  --output-csv generated/gan_ckpt100_fixed.csv
```

#### 📥 Fetch example data

The example datasets and pretrained checkpoint under `data/`, `generated/`, and `rawdata/` are
not committed to this repository (to keep it small) — they're downloaded on demand from
[Zenodo](https://zenodo.org/records/7157412) and this repo's release assets, streamed straight
to disk and checksum-verified.

```bash
uncorrupt-smiles fetch-data
```

Already-present files are skipped by default (pass `--force` to redownload). Use `--only` to
fetch specific files, e.g. `uncorrupt-smiles fetch-data --only rawdata/gbd_8.csv`.

#### 🧹 Standardize

Canonicalizes SMILES via RDKit. Anything RDKit can't parse — or an empty string — is dropped,
not silently kept.

```bash
uncorrupt-smiles standardize \
  --input-csv rawdata/PAPYRUS.csv --smiles-col SMILES --separator ";" \
  --output-csv data/papyrus_std.csv --threshold 200
```

`--smiles-col` and `--separator` are explicit arguments, so this works on any CSV — not just
PAPYRUS-shaped exports.

#### 🧪 Generate errors

Introduces synthetic, RDKit-confirmed-invalid corruptions of valid SMILES, for training or
evaluating a corrector.

```bash
uncorrupt-smiles generate-errors \
  --input-csv data/papyrus_std.csv --smiles-col STD_SMILES \
  --fragment-csv rawdata/gbd_8.csv --fragment-col FRAGMENT \
  --train-csv data/errors/train.csv --dev-csv data/errors/dev.csv \
  --invalid-type all --num-errors 1
```

`--invalid-type` selects which corruption(s) to introduce: `all`, `multiple`, or one of `exists`,
`par`, `permut`, `ring`, `syntax`, `valence`, `arom`.

#### 🏋️ Train

Training, evaluation, and checkpointing all live as methods on the model itself.

```bash
uncorrupt-smiles train \
  --train-csv data/errors/train.csv --dev-csv data/errors/dev.csv \
  --checkpoint-out data/performance/model.pkg \
  --hid-dim 128 --n-layers 2 --n-heads 4 --pf-dim 256 --batch-size 32 --epochs 20
```

The defaults above are sized for an 8GB GPU. For a full-scale run on more powerful hardware,
just pass larger values — e.g. `--hid-dim 512 --n-heads 8 --pf-dim 1024 --batch-size 128`.
`--device` defaults to CUDA if available, otherwise CPU. Pass `--resume <checkpoint>` to continue
training an existing checkpoint.

### Python

#### 🩹 Fix

```python
from uncorrupt_smiles import Uncorrupt

uncorrupt = Uncorrupt()  # or Uncorrupt(checkpoint="data/performance/model.pkg", device="cpu")

# a single SMILES, or a list
fixed = uncorrupt.fix_smiles(["C1=CC=CC=C1(", "CC(=O)Oc1ccccc1C(=O)O"])

# or stream an entire CSV in and out
uncorrupt.fix_smiles_csv(
    "generated/gan_ckpt100.csv", "SMILES", "generated/gan_ckpt100_fixed.csv",
)
```

`Uncorrupt` is a thin wrapper around `Seq2Seq`, the model class itself — reach for it directly
if you're already holding a model (e.g. right after `fit()`, or loaded via
`Seq2Seq.load_checkpoint(path, device)`), since a checkpoint's `src_vocab`/`trg_vocab` are
attached to the model, not passed around separately:

```python
from uncorrupt_smiles.transformer import Seq2Seq

model = Seq2Seq.load_checkpoint("data/performance/model.pkg", device="cpu")
model.fix_smiles(["C1=CC=CC=C1("])
```

`fix_smiles`/`fix_smiles_csv` work with a checkpoint trained by *this* model architecture, or any
future architecture exposing the same `generate(...)` method — nothing here is
Transformer-specific beyond `Seq2Seq` itself.

#### 📥 Fetch example data

```python
from uncorrupt_smiles.fetch_data import fetch_all

fetch_all()
```

Already-present files are skipped by default (pass `force=True` to redownload). Use
`only=[...]` to fetch specific files, e.g. `fetch_all(only=["rawdata/gbd_8.csv"])`.

#### 🧹 Standardize

```python
from uncorrupt_smiles.preprocess import standardize_smiles, standardize_stream

standardize_smiles("C(C)O")  # -> "CCO"

standardize_stream(
    "rawdata/PAPYRUS.csv", smiles_col="SMILES", output_csv="data/papyrus_std.csv",
    length_threshold=200, separator=";",
)
```

#### 🧪 Generate errors

```python
from uncorrupt_smiles.invalid_smiles import generate_errors, reservoir_sample_fragments, write_errors_split

# stream straight to train/dev CSVs
write_errors_split(
    "data/papyrus_std.csv", "STD_SMILES",
    "rawdata/gbd_8.csv", "FRAGMENT",
    "data/errors/train.csv", "data/errors/dev.csv",
    seed=42, invalid_type="all",
)

# or generate pairs in memory, e.g. for a handful of molecules
fragments = reservoir_sample_fragments("rawdata/gbd_8.csv", "FRAGMENT", k=20_000, seed=42)
pairs = list(generate_errors(["CCO", "c1ccccc1"], fragments, seed=42))
```

#### 🏋️ Train

```python
from uncorrupt_smiles.data import iter_csv_column, make_loader
from uncorrupt_smiles.transformer import Seq2Seq
from uncorrupt_smiles.vocab import Vocab

src_vocab = Vocab.build_from_lines(iter_csv_column("data/errors/train.csv", "ERROR"))
trg_vocab = Vocab.build_from_lines(iter_csv_column("data/errors/train.csv", "STD_SMILES"))

model = Seq2Seq.build(
    len(src_vocab), len(trg_vocab), max_length=202, device="cuda",
    src_pad_idx=src_vocab.pad_idx, trg_pad_idx=trg_vocab.pad_idx,
    hid_dim=128, n_layers=2, n_heads=4, pf_dim=256,
)

train_loader = make_loader("data/errors/train.csv", "ERROR", "STD_SMILES", src_vocab, trg_vocab, batch_size=32)
dev_loader = make_loader("data/errors/dev.csv", "ERROR", "STD_SMILES", src_vocab, trg_vocab, batch_size=32)

model.fit(train_loader, dev_loader, src_vocab, trg_vocab, epochs=20, checkpoint_path="data/performance/model.pkg")
```

The defaults above are sized for an 8GB GPU; pass larger `hid_dim`/`n_heads`/`pf_dim`/batch size
keyword arguments for a full-scale run on more powerful hardware. Use
`Seq2Seq.load_checkpoint(...)` to continue training an existing checkpoint.

## 🗂️ Package layout

| Module | Purpose |
|---|---|
| `src/uncorrupt_smiles/uncorrupt.py` | `Uncorrupt` — the main entrypoint. Wraps `Seq2Seq.load_checkpoint`, defaulting to the bundled pretrained checkpoint (downloaded on demand). |
| `src/uncorrupt_smiles/transformer.py` | The `Seq2Seq` model. Fixing SMILES (`fix_smiles`/`fix_smiles_csv`), training, evaluation, and checkpoint save/load are all methods on the model itself; `src_vocab`/`trg_vocab` are attached to the model, not passed around separately. |
| `src/uncorrupt_smiles/fetch_data.py` | Downloads example datasets / the pretrained checkpoint on demand. |
| `src/uncorrupt_smiles/invalid_smiles.py` | Per-SMILES error functions (`exists_error`, `par_error`, `permutation`, `ring_error`, `syntax_error`, `valence_error`, `arom_error`, `introduce_error`) plus streaming orchestration (`generate_errors`, `write_errors_split`, `reservoir_sample_fragments`). |
| `src/uncorrupt_smiles/preprocess.py` | `standardize_smiles` / `standardize_stream`. |
| `src/uncorrupt_smiles/vocab.py` | `Vocab` — character-level vocabulary, built by streaming. |
| `src/uncorrupt_smiles/data.py` | Streaming CSV helpers and the training `IterableDataset`. |
| `src/uncorrupt_smiles/cli.py` | The `uncorrupt-smiles` entrypoint. |

## ✅ Testing

```bash
pip install -e ".[test]"
pytest
```

The suite uses small synthetic fixtures and tiny model dimensions — it runs in a few seconds
and comfortably fits an 8GB-VRAM/modest-CPU machine.

## 🙏 Acknowledgments

uncorrupt-smiles builds on the original SMILES-corrector research and codebase by
Linde Schoenmaker.

## 📄 License

MIT — see [LICENSE](LICENSE).
