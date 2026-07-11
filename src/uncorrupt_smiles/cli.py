"""Command-line entrypoint: standardize / generate-errors / train / fix.

Everything is an explicit argument - no config files. Run `uncorrupt-smiles <subcommand> -h`
for the full flag list of any subcommand.
"""
from __future__ import annotations

import argparse

import torch

from uncorrupt_smiles.data import iter_csv_column, make_loader
from uncorrupt_smiles.fetch_data import fetch_all
from uncorrupt_smiles.invalid_smiles import INVALID_TYPES, write_errors_split
from uncorrupt_smiles.preprocess import standardize_stream
from uncorrupt_smiles.transformer import Seq2Seq
from uncorrupt_smiles.vocab import Vocab


def resolve_device(device: str | None) -> str:
    """Resolves the compute device to run on.

    :param device: Explicit device string (e.g. ``"cpu"``, ``"cuda"``), or
        ``None`` to auto-detect.
    :return: `device` if given, otherwise ``"cuda"`` if available else
        ``"cpu"``.
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_standardize(args: argparse.Namespace) -> None:
    """Runs the ``standardize`` subcommand: canonicalizes SMILES via RDKit.

    :param args: Parsed ``standardize`` arguments; must provide
        ``input_csv`` (SMILES to canonicalize), ``smiles_col``,
        ``output_csv`` (destination for canonicalized SMILES), ``threshold``
        (max token length, rows above it are dropped) and ``separator``.
    :return: None
    """
    standardize_stream(
        args.input_csv, args.smiles_col, args.output_csv,
        length_threshold=args.threshold, separator=args.separator,
    )
    print(f"standardized SMILES written to {args.output_csv}")


def cmd_generate_errors(args: argparse.Namespace) -> None:
    """Runs the ``generate-errors`` subcommand: builds synthetic
    invalid/valid SMILES pairs and writes a train/dev split.

    :param args: Parsed ``generate-errors`` arguments; must provide
        ``input_csv``/``smiles_col`` (standardized SMILES to corrupt),
        ``fragment_csv``/``fragment_col`` (fragment pool used to build
        corruptions), ``train_csv``/``dev_csv`` (destinations for the
        resulting split), ``seed``, ``invalid_type``, ``num_errors``,
        ``fragment_pool_size``, ``threshold`` (max SMILES token length) and
        ``frac_train`` (train/dev split fraction).
    :return: None
    """
    write_errors_split(
        args.input_csv, args.smiles_col, args.fragment_csv, args.fragment_col,
        args.train_csv, args.dev_csv, seed=args.seed, invalid_type=args.invalid_type,
        num_errors=args.num_errors, fragment_pool_size=args.fragment_pool_size,
        length_threshold=args.threshold, frac_train=args.frac_train,
    )
    print(f"train pairs written to {args.train_csv}, dev pairs written to {args.dev_csv}")


def cmd_train(args: argparse.Namespace) -> None:
    """Runs the ``train`` subcommand: trains a new corrector model, or
    resumes training from an existing checkpoint.

    :param args: Parsed ``train`` arguments; must provide ``train_csv``/
        ``dev_csv`` (training/validation data), ``src_col``/``trg_col``
        (columns holding the invalid/valid SMILES), ``checkpoint_out``
        (where to write the best checkpoint), ``resume`` (path to an
        existing checkpoint to continue from, or ``None`` to train from
        scratch), model hyperparameters (``vocab_max_size``, ``max_length``,
        ``hid_dim``, ``n_layers``, ``n_heads``, ``pf_dim``, ``dropout``),
        and training settings (``device``, ``seed``, ``batch_size``,
        ``epochs``, ``lr``, ``clip``, ``patience``, ``num_workers``,
        ``shuffle_buffer``).
    :return: None
    """
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    if args.resume:
        model = Seq2Seq.load_checkpoint(args.resume, device)
        src_vocab, trg_vocab = model.src_vocab, model.trg_vocab
        print(f"resumed model from {args.resume}: {model.hyperparams}")
    else:
        src_vocab = Vocab.build_from_lines(
            iter_csv_column(args.train_csv, args.src_col), max_size=args.vocab_max_size
        )
        trg_vocab = Vocab.build_from_lines(
            iter_csv_column(args.train_csv, args.trg_col), max_size=args.vocab_max_size
        )
        model = Seq2Seq.build(
            len(src_vocab), len(trg_vocab), args.max_length, device,
            src_vocab.pad_idx, trg_vocab.pad_idx,
            hid_dim=args.hid_dim, n_layers=args.n_layers, n_heads=args.n_heads,
            pf_dim=args.pf_dim, dropout=args.dropout,
        )

    train_loader = make_loader(
        args.train_csv, args.src_col, args.trg_col, src_vocab, trg_vocab,
        batch_size=args.batch_size, shuffle_buffer=args.shuffle_buffer,
        num_workers=args.num_workers, seed=args.seed,
    )
    dev_loader = None
    if args.dev_csv:
        dev_loader = make_loader(
            args.dev_csv, args.src_col, args.trg_col, src_vocab, trg_vocab,
            batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        )

    model.fit(
        train_loader, dev_loader, src_vocab, trg_vocab,
        epochs=args.epochs, lr=args.lr, clip=args.clip,
        checkpoint_path=args.checkpoint_out, patience=args.patience,
    )
    print(f"done, checkpoint at {args.checkpoint_out}")


def cmd_fetch_data(args: argparse.Namespace) -> None:
    """Runs the ``fetch-data`` subcommand: downloads bundled example
    datasets and/or the pretrained checkpoint.

    :param args: Parsed ``fetch-data`` arguments; must provide ``dest``
        (repo root to download into), ``only`` (restrict to these dest
        paths, or ``None`` for everything) and ``force`` (redownload even
        if a file already exists).
    :return: None
    """
    fetch_all(root=args.dest, only=args.only, force=args.force)


def cmd_fix(args: argparse.Namespace) -> None:
    """Runs the ``fix`` subcommand: corrects SMILES in a CSV using a trained
    checkpoint.

    :param args: Parsed ``fix`` arguments; must provide ``checkpoint``
        (trained model to load), ``input_csv``/``smiles_col`` (SMILES to
        correct), ``output_csv`` (destination for corrected SMILES),
        ``batch_size`` and ``device``.
    :return: None
    """
    device = resolve_device(args.device)
    model = Seq2Seq.load_checkpoint(args.checkpoint, device)
    model.fix_smiles_csv(
        args.input_csv, args.smiles_col, args.output_csv, batch_size=args.batch_size,
    )
    print(f"fixed SMILES written to {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argument parser and its subcommands.

    :return: Parser with ``standardize``, ``generate-errors``, ``train``,
        ``fetch-data``, and ``fix`` subcommands registered, each with
        ``func`` set to its handler via :meth:`~argparse.ArgumentParser.set_defaults`.
    """
    parser = argparse.ArgumentParser(prog="uncorrupt-smiles")
    sub = parser.add_subparsers(dest="command", required=True)

    p_std = sub.add_parser("standardize", help="canonicalize SMILES via RDKit, dropping unparseable rows")
    p_std.add_argument("--input-csv", required=True)
    p_std.add_argument("--smiles-col", default="SMILES")
    p_std.add_argument("--output-csv", required=True)
    p_std.add_argument("--threshold", type=int, default=None, help="max SMILES token length")
    p_std.add_argument("--separator", default=",")
    p_std.set_defaults(func=cmd_standardize)

    p_err = sub.add_parser("generate-errors", help="generate synthetic invalid/valid SMILES pairs")
    p_err.add_argument("--input-csv", required=True, help="standardized SMILES, any dataset")
    p_err.add_argument("--smiles-col", default="STD_SMILES")
    p_err.add_argument("--fragment-csv", required=True)
    p_err.add_argument("--fragment-col", default="FRAGMENT")
    p_err.add_argument("--fragment-pool-size", type=int, default=20_000)
    p_err.add_argument("--train-csv", required=True)
    p_err.add_argument("--dev-csv", required=True)
    p_err.add_argument("--invalid-type", default="all", choices=INVALID_TYPES)
    p_err.add_argument("--num-errors", type=int, default=1)
    p_err.add_argument("--threshold", type=int, default=200, help="max SMILES token length")
    p_err.add_argument("--frac-train", type=float, default=0.9)
    p_err.add_argument("--seed", type=int, default=42)
    p_err.set_defaults(func=cmd_generate_errors)

    p_train = sub.add_parser("train", help="train (or resume) a corrector model")
    p_train.add_argument("--train-csv", required=True)
    p_train.add_argument("--dev-csv", default=None)
    p_train.add_argument("--src-col", default="ERROR")
    p_train.add_argument("--trg-col", default="STD_SMILES")
    p_train.add_argument("--checkpoint-out", required=True)
    p_train.add_argument("--resume", default=None, help="path to an existing checkpoint to continue training")
    p_train.add_argument("--vocab-max-size", type=int, default=200)
    p_train.add_argument("--max-length", type=int, default=202)
    p_train.add_argument("--hid-dim", type=int, default=128)
    p_train.add_argument("--n-layers", type=int, default=2)
    p_train.add_argument("--n-heads", type=int, default=4)
    p_train.add_argument("--pf-dim", type=int, default=256)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--clip", type=float, default=0.1)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p_train.add_argument("--num-workers", type=int, default=0)
    p_train.add_argument("--shuffle-buffer", type=int, default=1000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_fetch = sub.add_parser("fetch-data", help="download bundled example datasets / pretrained checkpoint")
    p_fetch.add_argument("--dest", default=".", help="repo root to download into")
    p_fetch.add_argument("--only", nargs="*", default=None, help="only fetch these dest paths, e.g. rawdata/gbd_8.csv")
    p_fetch.add_argument("--force", action="store_true", help="redownload even if the file already exists")
    p_fetch.set_defaults(func=cmd_fetch_data)

    p_fix = sub.add_parser("fix", help="correct SMILES in a CSV using a trained checkpoint")
    p_fix.add_argument("--checkpoint", required=True)
    p_fix.add_argument("--input-csv", required=True)
    p_fix.add_argument("--smiles-col", default="SMILES")
    p_fix.add_argument("--output-csv", required=True)
    p_fix.add_argument("--batch-size", type=int, default=64)
    p_fix.add_argument("--device", default=None, help="default: cuda if available else cpu")
    p_fix.set_defaults(func=cmd_fix)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: parses arguments and dispatches to the selected
    subcommand handler.

    :param argv: Argument list to parse, excluding the program name; if
        ``None``, parses :data:`sys.argv` as usual.
    :return: None
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
