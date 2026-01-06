"""
Data readers for ETL pipeline.

Handles reading TSV files with various formats from MSK-IMPACT datasets.
"""

from pathlib import Path
from typing import Any

import pandas as pd


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

    # Read TSV with standard parameters
    # Use low_memory=False to avoid dtype warnings on large files with mixed types
    df = pd.read_csv(
        file_path,
        sep=sep,
        comment=comment,
        low_memory=False,
        **kwargs,
    )

    return df


def get_table_name(file_path: str | Path) -> str:
    """Extract table name from filename, removing 'data_' prefix if present."""
    file_path = Path(file_path)
    stem = file_path.stem

    # Remove 'data_' prefix if present
    if stem.startswith("data_"):
        return stem[5:]

    return stem
