"""Tests for hypothesis database module."""

from pathlib import Path

import pandas as pd  # type: ignore
import pytest

from geryon.db.database import Database, _get_table_name


def test_get_table_name_removes_data_prefix():
    assert _get_table_name(Path("data_clinical_patient.parquet")) == "clinical_patient"
    assert _get_table_name(Path("data_mutations.parquet")) == "mutations"


def test_get_table_name_handles_no_prefix():
    assert _get_table_name(Path("clinical_patient.parquet")) == "clinical_patient"
    assert _get_table_name(Path("mutations.parquet")) == "mutations"


def test_database_registers_parquet_files_as_tables(tmp_path: Path):
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

    db = Database(tmp_path)
    tables = db.list_tables()
    assert "clinical_patient" in tables
    db.close()


def test_database_with_nonexistent_directory_raises():
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        Database("/nonexistent/path")


def test_database_with_empty_directory_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="No parquet files found"):
        Database(tmp_path)


def test_database_executes_simple_query(tmp_path: Path):
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

    db = Database(tmp_path)
    result = db.execute("SELECT * FROM clinical_patient")

    assert len(result) == 3
    assert list(result.columns) == ["PATIENT_ID", "CANCER_TYPE"]
    db.close()


def test_database_executes_filtered_query(tmp_path: Path):
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

    db = Database(tmp_path)
    result = db.execute(
        "SELECT PATIENT_ID FROM clinical_patient "
        "WHERE CANCER_TYPE = 'Lung Adenocarcinoma'"
    )

    assert len(result) == 2
    assert result["PATIENT_ID"].tolist() == ["P001", "P003"]
    db.close()


def test_database_works_as_context_manager(tmp_path: Path):
    df = pd.DataFrame({"PATIENT_ID": ["P001"]})
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    with Database(tmp_path) as db:
        tables = db.list_tables()
        assert "clinical_patient" in tables


def test_database_loads_profiles(tmp_path: Path):
    """Profile JSON files alongside parquet are loaded into profiles dict."""
    import json

    df = pd.DataFrame({"PATIENT_ID": ["P001"]})
    df.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)

    profile = {"table_name": "clinical_patient", "row_count": 1, "columns": []}
    (tmp_path / "data_clinical_patient.profile.json").write_text(json.dumps(profile))

    db = Database(tmp_path)
    p = db.get_profile("clinical_patient")
    assert p is not None
    assert p["row_count"] == 1
    db.close()


def test_database_no_profiles_graceful(tmp_path: Path):
    """No profile files → empty dict, everything works."""
    df = pd.DataFrame({"PATIENT_ID": ["P001"]})
    df.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)

    db = Database(tmp_path)
    assert db.profiles == {}
    assert db.get_profile("clinical_patient") is None
    db.close()
