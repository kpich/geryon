"""Tests for ETL writers module."""

from pathlib import Path

import pandas as pd

from msk_cycl.etl.writers import write_parquet


def test_write_parquet_creates_output_directory(tmp_path: Path) -> None:
    """Verify that write_parquet creates parent directories if they don't exist."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    output_path = tmp_path / "nested" / "dir" / "output.parquet"

    write_parquet(df, output_path)

    assert output_path.exists()
    assert output_path.parent.exists()


def test_write_parquet_excludes_index_by_default(tmp_path: Path) -> None:
    """Verify that write_parquet does not include the index by default."""
    df = pd.DataFrame({"col1": [1, 2, 3]}, index=[10, 20, 30])
    output_path = tmp_path / "output.parquet"

    write_parquet(df, output_path)

    df_read = pd.read_parquet(output_path)
    # The index should be a default RangeIndex, not [10, 20, 30]
    assert list(df_read.index) == [0, 1, 2]


def test_write_parquet_respects_compression_parameter(tmp_path: Path) -> None:
    """Verify that write_parquet accepts different compression algorithms."""
    df = pd.DataFrame({"col1": range(100)})

    for compression in ["snappy", "gzip"]:
        output_path = tmp_path / f"output_{compression}.parquet"
        write_parquet(df, output_path, compression=compression)
        assert output_path.exists()


def test_write_parquet_strips_whitespace_from_string_columns(tmp_path: Path) -> None:
    """Verify that write_parquet strips trailing whitespace from string values."""
    df = pd.DataFrame(
        {
            "col1": ["value1   ", "value2  ", "value3"],
            "col2": [1, 2, 3],
            "col3": ["  leading", "  both  ", "normal"],
        }
    )
    output_path = tmp_path / "output.parquet"

    write_parquet(df, output_path)

    df_read = pd.read_parquet(output_path)
    # Trailing whitespace should be stripped
    assert df_read["col1"].tolist() == ["value1", "value2", "value3"]
    assert df_read["col3"].tolist() == ["leading", "both", "normal"]
    # Numeric columns should be unchanged
    assert df_read["col2"].tolist() == [1, 2, 3]
