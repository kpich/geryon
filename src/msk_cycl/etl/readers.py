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
    """
    Read a TSV file into a pandas DataFrame.

    TSV files from MSK-IMPACT may have:
    - Tab-separated values (despite .txt extension)
    - Comment lines starting with '#'
    - Various encodings

    Parameters
    ----------
    file_path : str or Path
        Path to the TSV file to read
    sep : str, default="\t"
        Field separator
    comment : str, default="#"
        Character indicating comment lines to skip
    **kwargs
        Additional arguments passed to pandas.read_csv

    Returns
    -------
    pd.DataFrame
        Parsed data from TSV file

    Raises
    ------
    FileNotFoundError
        If the file does not exist

    Examples
    --------
    >>> df = read_tsv("data_clinical_patient.txt")
    >>> df = read_tsv("data_mutations.txt", low_memory=False)

    Notes
    -----
    This is a boilerplate implementation. Schema validation and
    type coercion will be added in future iterations.
    """
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
    """
    Extract table name from file path.

    Converts file paths like:
    - "data_clinical_patient.txt" -> "clinical_patient"
    - "data_mutations_extended.txt" -> "mutations_extended"

    Parameters
    ----------
    file_path : str or Path
        Path to the data file

    Returns
    -------
    str
        Inferred table name

    Examples
    --------
    >>> get_table_name("data_clinical_patient.txt")
    'clinical_patient'
    >>> get_table_name("/path/to/data_mutations.txt")
    'mutations'
    """
    file_path = Path(file_path)
    stem = file_path.stem

    # Remove 'data_' prefix if present
    if stem.startswith("data_"):
        return stem[5:]

    return stem
