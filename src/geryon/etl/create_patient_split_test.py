"""Tests for patient split ETL script."""

from pathlib import Path

import pandas as pd  # type: ignore

from geryon.etl.create_patient_split import create_split


def test_create_split_writes_correct_columns(tmp_path: Path):
    input_path = tmp_path / "data_clinical_patient.parquet"
    pd.DataFrame({"PATIENT_ID": [f"P{i:03d}" for i in range(10)]}).to_parquet(
        input_path, index=False
    )
    output_path = tmp_path / "patient_split.parquet"

    create_split(str(input_path), str(output_path), seed=42)

    df = pd.read_parquet(output_path)
    assert set(df.columns) == {"PATIENT_ID", "split"}
    assert set(df["split"].unique()) <= {"explore", "validation"}


def test_create_split_roughly_80_20(tmp_path: Path):
    n = 100
    input_path = tmp_path / "data_clinical_patient.parquet"
    pd.DataFrame({"PATIENT_ID": [f"P{i:03d}" for i in range(n)]}).to_parquet(
        input_path, index=False
    )
    output_path = tmp_path / "patient_split.parquet"

    create_split(str(input_path), str(output_path), seed=42)

    df = pd.read_parquet(output_path)
    counts = df["split"].value_counts()
    assert counts["validation"] == round(n * 0.2)
    assert counts["explore"] == n - round(n * 0.2)


def test_create_split_is_stable_across_seeds(tmp_path: Path):
    input_path = tmp_path / "data_clinical_patient.parquet"
    pd.DataFrame({"PATIENT_ID": [f"P{i:03d}" for i in range(50)]}).to_parquet(
        input_path, index=False
    )
    out1 = tmp_path / "split1.parquet"
    out2 = tmp_path / "split2.parquet"

    create_split(str(input_path), str(out1), seed=42)
    create_split(str(input_path), str(out2), seed=42)

    df1 = pd.read_parquet(out1).sort_values("PATIENT_ID").reset_index(drop=True)
    df2 = pd.read_parquet(out2).sort_values("PATIENT_ID").reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df2)


def test_create_split_covers_all_patients(tmp_path: Path):
    patients = [f"P{i:03d}" for i in range(20)]
    input_path = tmp_path / "data_clinical_patient.parquet"
    pd.DataFrame({"PATIENT_ID": patients}).to_parquet(input_path, index=False)
    output_path = tmp_path / "patient_split.parquet"

    create_split(str(input_path), str(output_path), seed=7)

    df = pd.read_parquet(output_path)
    assert set(df["PATIENT_ID"]) == set(patients)
    assert len(df) == len(patients)
