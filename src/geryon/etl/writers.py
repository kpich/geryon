"""Parquet and profile writers for the ETL."""

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns to avoid parquet type conversion errors."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
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

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean = _clean_dataframe_for_parquet(df)

    df_clean.to_parquet(
        path=output_path,
        engine="pyarrow",
        compression=compression,
        index=index,
        **kwargs,
    )


def write_profile(profile: dict, output_path: Path) -> None:
    """Write column profile to JSON sidecar file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2))
