# 🧬 uncorrupt-smiles

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**uncorrupt-smiles** trains and applies Transformer models that translate invalid SMILES
sequences into valid ones. It is built around three composable, argument-driven building
blocks — no config files required:

1. **Generate errors** from any SMILES dataset.
2. **Train** a corrector model, whatever the hardware.
3. **Fix** SMILES with a trained model, from the CLI or from Python.

Every stage streams its input through [polars](https://pola.rs); no dataset is ever loaded
fully into memory, so the same commands scale from a laptop GPU to a training cluster.

## 📦 Installation

Requires Python 3.11+.

```bash
git clone https://github.com/OlivierBeq/uncorrupt-smiles.git
cd uncorrupt-smiles
pip install -e .
```

Optional extras:

```bash
pip install -e ".[analysis]"   # pandas/matplotlib/seaborn for the analysis/ scripts
pip install -e ".[test]"       # pytest, for running the test suite
```

Core dependencies — `torch >= 2.4`, `rdkit >= 2024.3.1`, `polars >= 1.0` — are resolved by pip
to whichever wheel (CPU or CUDA) matches your machine.

> **Note:** checkpoints produced by earlier versions of this project (including the archive on
> [Zenodo](https://zenodo.org/record/7157412#.Y1edr3ZBxD8)) use a different, incompatible
> checkpoint format and cannot be loaded by `Seq2Seq.load_checkpoint`. Retrain to obtain a
> checkpoint this version can use.

## 🚀 Usage

Every operation is available two ways: as a CLI subcommand, and as a plain Python function or
model method. Run `uncorrupt-smiles <subcommand> -h` for the full flag list of any subcommand.

### 📥 0. Fetch example data

The example datasets and pretrained checkpoint under `data/`, `generated/`, and `rawdata/` are
not committed to this repository (to keep it small) — they're downloaded on demand from
[Zenodo](https://zenodo.org/records/7157412) and this repo's release assets, streamed straight
to disk and checksum-verified.

**CLI**

```bash
uncorrupt-smiles fetch-data
```

**Python**

```python
from uncorrupt_smiles.fetch_data import fetch_all

fetch_all()
```

Already-present files are skipped by default (pass `--force`/`force=True` to redownload).
Use `--only`/`only=[...]` to fetch specific files, e.g.
`uncorrupt-smiles fetch-data --only rawdata/gbd_8.csv`.

### 🧹 1. Standardize

Canonicalizes SMILES via RDKit. Anything RDKit can't parse — or an empty string — is dropped,
not silently kept.

**CLI**

```bash
uncorrupt-smiles standardize \
  --input-csv rawdata/PAPYRUS.csv --smiles-col SMILES --separator ";" \
  --output-csv data/papyrus_std.csv --threshold 200
```

**Python**

```python
from uncorrupt_smiles.preprocess import standardize_smiles, standardize_stream

standardize_smiles("C(C)O")  # -> "CCO"

standardize_stream(
    "rawdata/PAPYRUS.csv", smiles_col="SMILES", output_csv="data/papyrus_std.csv",
    length_threshold=200, separator=";",
)
```

`--smiles-col`/`smiles_col` and `--separator`/`separator` are explicit arguments, so this works
on any CSV — not just PAPYRUS-shaped exports.

### 🧪 2. Generate errors

Introduces synthetic, RDKit-confirmed-invalid corruptions of valid SMILES, for training or
evaluating a corrector.

**CLI**

```bash
uncorrupt-smiles generate-errors \
  --input-csv data/papyrus_std.csv --smiles-col STD_SMILES \
  --fragment-csv rawdata/gbd_8.csv --fragment-col FRAGMENT \
  --train-csv data/errors/train.csv --dev-csv data/errors/dev.csv \
  --invalid-type all --num-errors 1
```

**Python**

```python
from uncorrupt_smiles.invalidSMILES import generate_errors, reservoir_sample_fragments, write_errors_split

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

`--invalid-type` (`invalid_type`) selects which corruption(s) to introduce: `all`, `multiple`,
or one of `exists`, `par`, `permut`, `ring`, `syntax`, `valence`, `arom`.

### 🏋️ 3. Train

Training, evaluation, and checkpointing all live as methods on the model itself.

**CLI**

```bash
uncorrupt-smiles train \
  --train-csv data/errors/train.csv --dev-csv data/errors/dev.csv \
  --checkpoint-out data/performance/model.pkg \
  --hid-dim 128 --n-layers 2 --n-heads 4 --pf-dim 256 --batch-size 32 --epochs 20
```

**Python**

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

The defaults above are sized for an 8GB GPU. For a full-scale run on more powerful hardware,
just pass larger values — e.g. `--hid-dim 512 --n-heads 8 --pf-dim 1024 --batch-size 128`
(or the matching keyword arguments in Python). `--device`/`device` defaults to CUDA if
available, otherwise CPU. Pass `--resume <checkpoint>` (CLI) or use
`Seq2Seq.load_checkpoint(...)` (Python) to continue training an existing checkpoint.

### 🩹 4. Fix

**CLI**

```bash
uncorrupt-smiles fix \
  --checkpoint data/performance/model.pkg \
  --input-csv generated/gan_ckpt100.csv --smiles-col SMILES \
  --output-csv generated/gan_ckpt100_fixed.csv
```

**Python**

```python
from uncorrupt_smiles.transformer import Seq2Seq

model, src_vocab, trg_vocab = Seq2Seq.load_checkpoint("data/performance/model.pkg", device="cpu")

# a single SMILES, or a list
fixed = model.fix_smiles(
    ["C1=CC=CC=C1(", "CC(=O)Oc1ccccc1C(=O)O"], src_vocab, trg_vocab,
)

# or stream an entire CSV in and out
model.fix_smiles_csv(
    "generated/gan_ckpt100.csv", "SMILES", "generated/gan_ckpt100_fixed.csv",
    src_vocab, trg_vocab,
)
```

`fix_smiles`/`fix_smiles_csv` work with a checkpoint trained by *this* model architecture, or
any future architecture exposing the same `generate(...)` method — nothing here is
Transformer-specific beyond `Seq2Seq` itself.

## 🗂️ Package layout

| Module | Purpose |
|---|---|
| `src/uncorrupt_smiles/invalidSMILES.py` | Per-SMILES error functions (`exists_error`, `par_error`, `permutation`, `ring_error`, `syntax_error`, `valence_error`, `arom_error`, `introduce_error`) plus streaming orchestration (`generate_errors`, `write_errors_split`, `reservoir_sample_fragments`). |
| `src/uncorrupt_smiles/preprocess.py` | `standardize_smiles` / `standardize_stream`. |
| `src/uncorrupt_smiles/vocab.py` | `Vocab` — character-level vocabulary, built by streaming. |
| `src/uncorrupt_smiles/data.py` | Streaming CSV helpers and the training `IterableDataset`. |
| `src/uncorrupt_smiles/transformer.py` | The `Seq2Seq` model. Training, evaluation, checkpoint save/load, and `fix_smiles`/`fix_smiles_csv` are all methods on the model itself. |
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
