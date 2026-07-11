"""Uncorrupt: the main, batteries-included entrypoint for fixing SMILES.

Wraps Seq2Seq.load_checkpoint so that just fixing SMILES needs no explicit checkpoint path,
vocab handling, or device selection - by default it loads (downloading it first if necessary)
the pretrained checkpoint bundled with this project.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from uncorrupt_smiles.fetch_data import DEFAULT_CHECKPOINT, fetch_all
from uncorrupt_smiles.transformer import Seq2Seq


class Uncorrupt:
    """Loads a trained corrector checkpoint and fixes SMILES with it.

    With no arguments, downloads (if not already present) and loads the pretrained checkpoint
    bundled with this project - this is the main way to use uncorrupt-smiles:

        from uncorrupt_smiles import Uncorrupt

        uncorrupt = Uncorrupt()
        uncorrupt.fix_smiles("C1=CC=CC=C1(")  # -> ["c1ccccc1"]

    Pass `checkpoint` to use a different (e.g. self-trained) checkpoint instead.
    """

    def __init__(self, checkpoint: str | None = None, device: str | None = None):
        """
        :param checkpoint: Path to a checkpoint written by
            :meth:`~uncorrupt_smiles.transformer.Seq2Seq.save_checkpoint`. If
            ``None``, uses the pretrained checkpoint bundled with this
            project, downloading it first if not already present.
        :param device: Torch device string to load the model onto. If
            ``None``, uses ``"cuda"`` when available, otherwise ``"cpu"``.
        """
        if checkpoint is None:
            checkpoint = DEFAULT_CHECKPOINT
            if not Path(checkpoint).exists():
                fetch_all(only=[DEFAULT_CHECKPOINT])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint = checkpoint
        self.device = device
        self.model: Seq2Seq = Seq2Seq.load_checkpoint(checkpoint, device)

    def fix_smiles(
        self,
        smiles: str | Iterable[str],
        max_len: int | None = None,
        batch_size: int = 64,
    ) -> list[str]:
        """Fixes a single SMILES string or an iterable of them.

        :param smiles: A single (possibly invalid) SMILES string, or an
            iterable of them, to correct.
        :param max_len: Maximum number of tokens to generate per corrected
            SMILES. If ``None``, uses the model's default.
        :param batch_size: Number of SMILES translated per forward pass
            through the model.
        :return: Corrected SMILES, in the same order as `smiles`.
        """
        return self.model.fix_smiles(smiles, max_len=max_len, batch_size=batch_size)

    def fix_smiles_csv(
        self,
        input_csv: str,
        smiles_col: str,
        output_csv: str,
        batch_size: int = 64,
        separator: str = ",",
    ) -> None:
        """Streams an entire CSV of SMILES through :meth:`fix_smiles`, writing
        results incrementally.

        :param input_csv: Path to the CSV file containing the SMILES to
            correct.
        :param smiles_col: Name of the column in `input_csv` holding the
            SMILES to correct.
        :param output_csv: Path to write the corrected SMILES to, as a new
            CSV file.
        :param batch_size: Number of SMILES translated per forward pass
            through the model.
        :param separator: Field separator used by both `input_csv` and
            `output_csv`.
        """
        self.model.fix_smiles_csv(
            input_csv, smiles_col, output_csv, batch_size=batch_size, separator=separator,
        )
