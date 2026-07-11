"""Dataset standardization: canonicalize SMILES and drop anything RDKit can't parse.

No chembl_structure_pipeline dependency - RDKit alone is the standardization authority. This
means salts/multi-fragment SMILES are left as RDKit canonicalizes them (no desalting/largest-
fragment selection); anything RDKit can't parse at all is simply dropped.
"""
from __future__ import annotations

from typing import Any

import polars as pl
from rdkit import Chem

from uncorrupt_smiles.utils.chem import is_valid_smiles
from uncorrupt_smiles.utils.tokenizer import smi_tokenizer


def standardize_smiles(smiles: Any) -> str | None:
    """Parses via RDKit and re-emits the canonical SMILES.

    This is the sole "standardization" step: no desalting or largest-fragment
    selection is performed, so salts/multi-fragment SMILES are canonicalized
    as-is.

    :param smiles: Value to standardize; anything :func:`~uncorrupt_smiles.utils.chem.is_valid_smiles`
        rejects is treated as unparseable.
    :return: Canonical SMILES, or ``None`` if RDKit can't parse the input, or
        it's empty (caller drops the row in that case).
    """
    if not is_valid_smiles(smiles):
        return None
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


def _token_length(smiles: str) -> int:
    """Counts SMILES tokens (not characters).

    :param smiles: SMILES string to tokenize via :func:`~uncorrupt_smiles.utils.tokenizer.smi_tokenizer`.
    :return: Number of tokens.
    """
    return len(smi_tokenizer(smiles))


def standardize_stream(
    input_csv: str,
    smiles_col: str,
    output_csv: str,
    length_threshold: int | None = None,
    separator: str = ",",
) -> None:
    """Streams ``input_csv`` -> RDKit-standardize -> drop unparseable/duplicate/(optionally)
    too-long rows -> ``output_csv``, via polars' lazy engine so the full dataset
    is never held in RAM at once.

    Note on memory: true de-duplication requires remembering every unique
    canonical SMILES seen so far (one string per unique molecule) - that is an
    inherent floor for this operation, not a full-dataset load, but it does
    mean memory scales with the number of *unique* output molecules rather
    than staying flat.

    :param input_csv: Path to the raw input CSV to read SMILES from.
    :param smiles_col: Name of the column in ``input_csv`` holding SMILES to
        standardize.
    :param output_csv: Destination path to write the standardized,
        deduplicated CSV to.
    :param length_threshold: If set, drop rows whose standardized SMILES
        exceeds this many tokens (via :func:`_token_length`, i.e.
        :func:`~uncorrupt_smiles.utils.tokenizer.smi_tokenizer` token count,
        matching how downstream error-generation measures sequence length -
        not raw character count).
    :param separator: Field separator of ``input_csv`` (explicit rather than
        hardcoded so this works on any dataset's delimiter, e.g.
        ``rawdata/PAPYRUS.csv`` uses ``";"``).
    """
    # Force the SMILES column to Utf8: a column that happens to contain only digit-looking
    # values would otherwise be silently inferred as numeric by polars.
    lf = pl.scan_csv(input_csv, separator=separator, schema_overrides={smiles_col: pl.Utf8})
    lf = lf.select(
        pl.col(smiles_col)
        .map_elements(standardize_smiles, return_dtype=pl.Utf8)
        .alias("STD_SMILES")
    ).filter(pl.col("STD_SMILES").is_not_null())

    if length_threshold is not None:
        lf = lf.filter(
            pl.col("STD_SMILES")
            .map_elements(_token_length, return_dtype=pl.Int64)
            .le(length_threshold)
        )

    lf = lf.unique(subset=["STD_SMILES"], keep="first")
    lf.sink_csv(output_csv)
