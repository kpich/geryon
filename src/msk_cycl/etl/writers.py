"""
Data writers for ETL pipeline.

Handles writing DataFrames to parquet format for efficient storage and querying.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    compression: str = "snappy",
    index: bool = False,
    **kwargs: Any,
) -> None:
    """
    Write a DataFrame to parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write
    output_path : str or Path
        Output file path (should end with .parquet)
    compression : str, default="snappy"
        Compression algorithm. Options: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd'
        'snappy' provides good balance of speed and compression
    index : bool, default=False
        Whether to include DataFrame index in output
    **kwargs
        Additional arguments passed to pandas.to_parquet

    Examples
    --------
    >>> df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    >>> write_parquet(df, "output.parquet")
    >>> write_parquet(df, "output.parquet", compression="gzip")

    Notes
    -----
    This is a boilerplate implementation. Future enhancements may include:
    - Schema enforcement
    - Partitioning for large datasets
    - Metadata embedding
    """
    output_path = Path(output_path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to parquet
    df.to_parquet(
        output_path,
        compression=compression,
        index=index,
        engine="pyarrow",  # Explicit engine for consistency
        **kwargs,
    )


def get_parquet_metadata(file_path: str | Path) -> dict[str, Any]:
    """
    Extract metadata from a parquet file.

    Useful for validation and debugging.

    Parameters
    ----------
    file_path : str or Path
        Path to parquet file

    Returns
    -------
    dict
        Metadata including schema, row count, etc.

    Raises
    ------
    FileNotFoundError
        If the file does not exist

    Examples
    --------
    >>> metadata = get_parquet_metadata("output.parquet")
    >>> print(f"Rows: {metadata['num_rows']}")
    """
    import pyarrow.parquet as pq

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    parquet_file = pq.ParquetFile(file_path)

    return {
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": parquet_file.metadata.num_columns,
        "num_row_groups": parquet_file.metadata.num_row_groups,
        "schema": parquet_file.schema.to_arrow_schema(),
        "created_by": parquet_file.metadata.created_by,
    }
