"""Tests for hypothesis database module."""

from pathlib import Path

import pandas as pd
import pytest

from msk_cycl.hypothesis.db import Database, _get_table_name


def test_get_table_name_removes_data_prefix() -> None:
    """Verify table name extraction removes 'data_' prefix."""
    assert _get_table_name(Path("data_clinical_patient.parquet")) == "clinical_patient"
    assert _get_table_name(Path("data_mutations.parquet")) == "mutations"


def test_get_table_name_handles_no_prefix() -> None:
    """Verify table name extraction handles files without prefix."""
    assert _get_table_name(Path("clinical_patient.parquet")) == "clinical_patient"
    assert _get_table_name(Path("mutations.parquet")) == "mutations"


def test_database_initializes_with_parquet_files(tmp_path: Path) -> None:
    """Verify Database connects and registers parquet files."""
    # Create test parquet file
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "CANCER_TYPE": [
                "Lung Adenocarcinoma",
                "Breast Cancer",
                "Lung Adenocarcinoma",
            ],
            "AGE": [65, 52, 70],
        }
    )
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    # Initialize database
    db = Database(tmp_path)

    # Verify table is registered
    tables = db.list_tables()
    assert "clinical_patient" in tables

    db.close()


def test_database_raises_for_nonexistent_directory() -> None:
    """Verify Database raises error for nonexistent directory."""
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        Database("/nonexistent/path")


def test_database_raises_for_empty_directory(tmp_path: Path) -> None:
    """Verify Database raises error for directory with no parquet files."""
    with pytest.raises(ValueError, match="No parquet files found"):
        Database(tmp_path)


def test_database_execute_simple_query(tmp_path: Path) -> None:
    """Verify Database can execute SQL queries."""
    # Create test parquet file
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "CANCER_TYPE": [
                "Lung Adenocarcinoma",
                "Breast Cancer",
                "Lung Adenocarcinoma",
            ],
        }
    )
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    # Initialize and query
    db = Database(tmp_path)
    result = db.execute("SELECT * FROM clinical_patient")

    assert len(result) == 3
    assert list(result.columns) == ["PATIENT_ID", "CANCER_TYPE"]

    db.close()


def test_database_execute_filtered_query(tmp_path: Path) -> None:
    """Verify Database handles WHERE clauses correctly."""
    # Create test parquet file
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "CANCER_TYPE": [
                "Lung Adenocarcinoma",
                "Breast Cancer",
                "Lung Adenocarcinoma",
            ],
        }
    )
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    # Initialize and query with filter
    db = Database(tmp_path)
    result = db.execute(
        "SELECT PATIENT_ID FROM clinical_patient "
        "WHERE CANCER_TYPE = 'Lung Adenocarcinoma'"
    )

    assert len(result) == 2
    assert result["PATIENT_ID"].tolist() == ["P001", "P003"]

    db.close()


def test_database_context_manager(tmp_path: Path) -> None:
    """Verify Database works as context manager."""
    # Create test parquet file
    df = pd.DataFrame({"PATIENT_ID": ["P001"]})
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    # Use as context manager
    with Database(tmp_path) as db:
        tables = db.list_tables()
        assert "clinical_patient" in tables

    # Connection should be closed after context
