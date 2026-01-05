"""Tests for ETL writers module."""

from pathlib import Path

import pandas as pd
import pytest

from msk_cycl.etl.writers import get_parquet_metadata, write_parquet


def test_write_parquet_creates_output_directory(tmp_path: Path) -> None:
    """Verify that write_parquet creates parent directories if they don't exist."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    output_path = tmp_path / "nested" / "dir" / "output.parquet"

    write_parquet(df, output_path)

    assert output_path.exists()
    assert output_path.parent.exists()


def test_write_parquet_creates_readable_file(tmp_path: Path) -> None:
    """Verify that write_parquet creates a parquet file readable by pandas."""
    df = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.5, 2.5, 3.5]}
    )
    output_path = tmp_path / "output.parquet"

    write_parquet(df, output_path)

    df_read = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(df, df_read)


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


def test_get_parquet_metadata_raises_filenotfound_for_nonexistent_file() -> None:
    """Verify that get_parquet_metadata raises FileNotFoundError for nonexistent
    files."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        get_parquet_metadata("/nonexistent/path/to/file.parquet")


def test_get_parquet_metadata_returns_correct_row_count(tmp_path: Path) -> None:
    """Verify that get_parquet_metadata correctly reports row count."""
    df = pd.DataFrame({"col1": range(42), "col2": range(42, 84)})
    output_path = tmp_path / "output.parquet"
    write_parquet(df, output_path)

    metadata = get_parquet_metadata(output_path)

    assert metadata["num_rows"] == 42
    assert metadata["num_columns"] == 2


def test_get_parquet_metadata_includes_schema_information(tmp_path: Path) -> None:
    """Verify that get_parquet_metadata includes schema information."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    output_path = tmp_path / "output.parquet"
    write_parquet(df, output_path)

    metadata = get_parquet_metadata(output_path)

    assert "schema" in metadata
    assert metadata["schema"] is not None
    # Check that schema contains our columns
    schema_str = str(metadata["schema"])
    assert "col1" in schema_str
    assert "col2" in schema_str
