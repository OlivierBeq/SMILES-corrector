"""Dataset standardization: canonicalize SMILES and drop anything RDKit can't parse.

No chembl_structure_pipeline dependency - RDKit alone is the standardization authority. This
means salts/multi-fragment SMILES are left as RDKit canonicalizes them (no desalting/largest-
fragment selection); anything RDKit can't parse at all is simply dropped.
"""
from __future__ import annotations

import polars as pl
from rdkit import Chem

from uncorrupt_smiles.utils.chem import is_valid_smiles
from uncorrupt_smiles.utils.tokenizer import smi_tokenizer


def standardize_smiles(smiles) -> str | None:
    """Parse via RDKit and re-emit the canonical SMILES. Returns None if RDKit can't
    parse the input, or it's empty (caller drops the row) - this is the sole
    "standardization" step."""
    if not is_valid_smiles(smiles):
        return None
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)


def _token_length(smiles: str) -> int:
    return len(smi_tokenizer(smiles))


def standardize_stream(
    input_csv: str,
    smiles_col: str,
    output_csv: str,
    length_threshold: int | None = None,
    separator: str = ",",
) -> None:
    """Streams input_csv -> RDKit-standardize -> drop unparseable/duplicate/(optionally)
    too-long rows -> output_csv, via polars' lazy engine so the full dataset is never held
    in RAM at once. `separator` is explicit (not hardcoded) so this works on any dataset's
    delimiter, e.g. rawdata/PAPYRUS.csv uses ";". `length_threshold` counts SMILES *tokens*
    (via smi_tokenizer), matching how downstream error-generation measures sequence length -
    not raw character count.

    Note on memory: true de-duplication requires remembering every unique canonical SMILES
    seen so far (one string per unique molecule) - that is an inherent floor for this
    operation, not a full-dataset load, but it does mean memory scales with the number of
    *unique* output molecules rather than staying flat.
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
