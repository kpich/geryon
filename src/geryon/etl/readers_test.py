"""Tests for ETL readers module."""

from pathlib import Path

from geryon.etl.readers import get_table_name, read_tsv


def test_read_tsv_handles_comments(tmp_path: Path) -> None:
    """Verify that read_tsv correctly skips comment lines starting with #."""
    test_file = tmp_path / "test_data.txt"
    content = """# This is a comment
# Another comment
col1\tcol2\tcol3
val1\tval2\tval3
val4\tval5\tval6
"""
    test_file.write_text(content)

    df = read_tsv(test_file)

    assert len(df) == 2
    assert list(df.columns) == ["col1", "col2", "col3"]
    assert df.iloc[0]["col1"] == "val1"


def test_get_table_name_removes_data_prefix() -> None:
    """Verify that get_table_name removes 'data_' prefix from filename."""
    assert get_table_name("data_clinical_patient.txt") == "clinical_patient"
    assert get_table_name("data_mutations.txt") == "mutations"
    assert get_table_name("data_mutations_extended.txt") == "mutations_extended"


def test_get_table_name_handles_no_prefix() -> None:
    """Verify that get_table_name handles files without 'data_' prefix."""
    assert get_table_name("clinical_patient.txt") == "clinical_patient"
    assert get_table_name("mutations.txt") == "mutations"


def test_get_table_name_handles_path_objects() -> None:
    """Verify that get_table_name works with Path objects."""
    path = Path("/some/dir/data_clinical_sample.txt")
    assert get_table_name(path) == "clinical_sample"
