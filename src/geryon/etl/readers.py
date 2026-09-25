"""Readers for the MSK-IMPACT cBioPortal TSV files."""

from pathlib import Path
from typing import Any

import pandas as pd

from geryon.etl.writers import write_parquet


def read_tsv(
    file_path: str | Path,
    *,
    sep: str = "\t",
    comment: str = "#",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read TSV file with comment line handling and low_memory=False for mixed types."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(
        file_path,
        sep=sep,
        comment=comment,
        low_memory=False,
        **kwargs,
    )


def write_cna_matrix_to_parquet(
    input_path: str | Path, output_path: str | Path
) -> None:
    """Transpose the CNA matrix to one row per sample, one column per gene.

    The output key column is named ``PATIENT_ID`` but holds sample barcodes (see
    ``SAMPLE_KEY_COLUMNS`` in split_by_patient).
    """
    # Reading as strings skips type inference across ~150k sample columns.
    df = pd.read_csv(input_path, sep="\t", dtype=str, low_memory=False)

    df = df.set_index(df.columns[0]).T
    df.index.name = "PATIENT_ID"
    df = df.reset_index()

    # Duplicate Hugo_Symbol rows become duplicate columns, which break pd.to_numeric.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    write_parquet(df, output_path)


def get_table_name(file_path: str | Path) -> str:
    """Extract table name from filename, removing 'data_' prefix if present."""
    stem = Path(file_path).stem
    if stem.startswith("data_"):
        return stem[5:]

    return stem
