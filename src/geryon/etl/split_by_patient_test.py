"""Tests for the data-layer patient split."""

from pathlib import Path

import pandas as pd  # type: ignore

from geryon.etl.split_by_patient import SPLIT_MARKER_FILENAME, split_directory


def _build_input(tmp_path: Path) -> Path:
    """An ETL-shaped input dir: patients, samples, a sample-keyed table, metadata."""
    d = tmp_path / "in"
    d.mkdir()

    # 4 patients, 2 explore / 2 validation
    pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2", "P3", "P4"],
            "split": ["explore", "explore", "validation", "validation"],
        }
    ).to_parquet(d / "patient_split.parquet", index=False)

    pd.DataFrame(
        {"PATIENT_ID": ["P1", "P2", "P3", "P4"], "OS_MONTHS": [1, 2, 3, 4]}
    ).to_parquet(d / "data_clinical_patient.parquet", index=False)

    # P1->S1, P2->S2, P3->S3, P4->S4; plus an orphan sample S9 (no patient)
    pd.DataFrame(
        {
            "SAMPLE_ID": ["S1", "S2", "S3", "S4"],
            "PATIENT_ID": ["P1", "P2", "P3", "P4"],
            "CANCER_TYPE": ["a", "b", "c", "d"],
        }
    ).to_parquet(d / "data_clinical_sample.parquet", index=False)

    # mutations keyed by Tumor_Sample_Barcode, including orphan S9
    pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1", "S2", "S3", "S4", "S9"],
            "Hugo_Symbol": ["TP53", "KRAS", "EGFR", "BRAF", "PTEN"],
        }
    ).to_parquet(d / "data_mutations_extended.parquet", index=False)

    # CNA: column is *named* PATIENT_ID but holds SAMPLE_IDs (the real-data trap)
    pd.DataFrame(
        {"PATIENT_ID": ["S1", "S2", "S3", "S4"], "TP53": [-1, 0, 1, 2]}
    ).to_parquet(d / "data_CNA.parquet", index=False)

    # metadata: no patient/sample key -> copied verbatim
    pd.DataFrame({"note": ["license text"]}).to_parquet(
        d / "meta_study.parquet", index=False
    )

    return d


def test_split_creates_subdir_per_split(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"

    written = split_directory(inp, out)

    names = {p.name for p in written}
    assert names == {"explore", "validation"}
    assert (out / "explore" / SPLIT_MARKER_FILENAME).read_text().strip() == "explore"
    assert (out / "validation" / SPLIT_MARKER_FILENAME).read_text().strip() == (
        "validation"
    )


def test_patient_table_filtered_to_split(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    explore = pd.read_parquet(out / "explore" / "data_clinical_patient.parquet")
    validation = pd.read_parquet(out / "validation" / "data_clinical_patient.parquet")
    assert set(explore["PATIENT_ID"]) == {"P1", "P2"}
    assert set(validation["PATIENT_ID"]) == {"P3", "P4"}


def test_sample_keyed_table_follows_patient_split(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    explore = pd.read_parquet(out / "explore" / "data_mutations_extended.parquet")
    validation = pd.read_parquet(out / "validation" / "data_mutations_extended.parquet")
    assert set(explore["Tumor_Sample_Barcode"]) == {"S1", "S2"}
    assert set(validation["Tumor_Sample_Barcode"]) == {"S3", "S4"}


def test_cna_misnamed_patient_id_is_sample_keyed(tmp_path: Path):
    """CNA's PATIENT_ID column holds SAMPLE_IDs; split by sample, not patient."""
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    explore = pd.read_parquet(out / "explore" / "data_CNA.parquet")
    validation = pd.read_parquet(out / "validation" / "data_CNA.parquet")
    # S1,S2 belong to explore patients P1,P2; S3,S4 to validation P3,P4
    assert set(explore["PATIENT_ID"]) == {"S1", "S2"}
    assert set(validation["PATIENT_ID"]) == {"S3", "S4"}


def test_no_patient_overlap_between_splits(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    explore = pd.read_parquet(out / "explore" / "data_clinical_patient.parquet")
    validation = pd.read_parquet(out / "validation" / "data_clinical_patient.parquet")
    assert set(explore["PATIENT_ID"]).isdisjoint(set(validation["PATIENT_ID"]))


def test_orphan_sample_dropped_from_all_splits(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    explore = pd.read_parquet(out / "explore" / "data_mutations_extended.parquet")
    validation = pd.read_parquet(out / "validation" / "data_mutations_extended.parquet")
    seen = set(explore["Tumor_Sample_Barcode"]) | set(
        validation["Tumor_Sample_Barcode"]
    )
    assert "S9" not in seen


def test_metadata_copied_verbatim(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    for split in ("explore", "validation"):
        meta = pd.read_parquet(out / split / "meta_study.parquet")
        assert meta["note"].tolist() == ["license text"]


def test_split_table_not_copied_into_splits(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    assert not (out / "explore" / "patient_split.parquet").exists()
    assert not (out / "validation" / "patient_split.parquet").exists()


def test_profiles_regenerated_for_filtered_tables(tmp_path: Path):
    inp = _build_input(tmp_path)
    out = tmp_path / "out"
    split_directory(inp, out)

    profile = out / "explore" / "data_clinical_patient.profile.json"
    assert profile.exists()
