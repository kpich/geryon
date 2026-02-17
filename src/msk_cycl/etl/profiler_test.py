"""Tests for column profiler."""

from pathlib import Path

import pandas as pd
import pytest

from msk_cycl.etl.profiler import profile_parquet


def _make_parquet(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    path = tmp_path / name
    df.to_parquet(path, index=False)
    return path


def test_profile_basic(tmp_path: Path):
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "CANCER_TYPE": ["Lung", "Breast", "Lung"],
            "AGE": [65, 52, 70],
        }
    )
    path = _make_parquet(tmp_path, "data_clinical_patient.parquet", df)

    profile = profile_parquet(path)

    assert profile["table_name"] == "clinical_patient"
    assert profile["row_count"] == 3
    assert profile["column_count"] == 3
    assert profile["profile_strategy"] == "standard"
    assert len(profile["columns"]) == 3

    col_names = [c["column_name"] for c in profile["columns"]]
    assert "PATIENT_ID" in col_names
    assert "CANCER_TYPE" in col_names
    assert "AGE" in col_names


def test_profile_low_cardinality_all_values(tmp_path: Path):
    """<=20 distinct values → all included in top_values."""
    values = [f"TYPE_{i}" for i in range(15)]
    df = pd.DataFrame({"CATEGORY": values})
    path = _make_parquet(tmp_path, "test.parquet", df)

    profile = profile_parquet(path)

    col = profile["columns"][0]
    assert col["distinct_count"] == 15
    assert col["is_high_cardinality"] is False
    assert len(col["top_values"]) == 15


def test_profile_high_cardinality_limits(tmp_path: Path):
    """>20 distinct → is_high_cardinality=True, capped at top_n."""
    values = [f"VAL_{i}" for i in range(50)]
    df = pd.DataFrame({"MANY_VALUES": values})
    path = _make_parquet(tmp_path, "test.parquet", df)

    profile = profile_parquet(path, top_n=10)

    col = profile["columns"][0]
    assert col["distinct_count"] == 50
    assert col["is_high_cardinality"] is True
    assert len(col["top_values"]) == 10


def test_profile_null_rate(tmp_path: Path):
    df = pd.DataFrame(
        {
            "COL": ["A", None, "B", None, "C"],
        }
    )
    path = _make_parquet(tmp_path, "test.parquet", df)

    profile = profile_parquet(path)

    col = profile["columns"][0]
    assert col["null_count"] == 2
    assert col["null_rate"] == pytest.approx(0.4)
    assert col["total_count"] == 5


def test_profile_wide_matrix_detection(tmp_path: Path):
    """200+ columns → strategy='wide_matrix', gene_columns populated."""
    data: dict[str, list] = {"PATIENT_ID": ["P001", "P002"]}
    gene_names = [f"GENE_{i}" for i in range(200)]
    for gene in gene_names:
        data[gene] = [0, 1]
    df = pd.DataFrame(data)
    path = _make_parquet(tmp_path, "data_gene_matrix.parquet", df)

    profile = profile_parquet(path)

    assert profile["profile_strategy"] == "wide_matrix"
    assert "gene_columns" in profile
    assert len(profile["gene_columns"]) == 200
    # PATIENT_ID + 10 representative genes
    assert len(profile["columns"]) == 11


def test_profile_empty_table(tmp_path: Path):
    df = pd.DataFrame(
        {"A": pd.Series([], dtype="str"), "B": pd.Series([], dtype="int64")}
    )
    path = _make_parquet(tmp_path, "empty.parquet", df)

    profile = profile_parquet(path)

    assert profile["row_count"] == 0
    assert profile["column_count"] == 2
    for col in profile["columns"]:
        assert col["null_count"] == 0
        assert col["null_rate"] == 0.0
        assert col["distinct_count"] == 0
        assert col["top_values"] == []
