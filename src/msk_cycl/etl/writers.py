"""
Data writers for ETL pipeline.

Handles writing DataFrames to parquet format for efficient storage and querying.
"""

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow.parquet as pq  # type: ignore


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns to avoid parquet type conversion errors."""
    df_clean = df.copy()

    # Strip whitespace from object/string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            # Only apply str methods to non-null values
            df_clean[col] = df_clean[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )

    return df_clean


def write_parquet(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
    index: bool = False,
    **kwargs: Any,
) -> None:
    """Write DataFrame to parquet, stripping whitespace from string columns first."""
    output_path = Path(output_path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean DataFrame (strip whitespace from string columns)
    df_clean = _clean_dataframe_for_parquet(df)

    # Write to parquet
    df_clean.to_parquet(
        path=output_path,
        engine="pyarrow",
        compression=compression,
        index=index,
        **kwargs,
    )


def get_parquet_metadata(file_path: str | Path) -> dict[str, Any]:
    """Extract parquet file metadata (row count, column count, schema, etc.)."""
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
