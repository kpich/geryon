"""Unit tests for create_derived_view."""

from unittest.mock import MagicMock

import pandas as pd

from geryon.tools.derived import create_derived_view


def _make_db(etl_tables=None, row_count=42, cols=None):
    db = MagicMock()
    db.list_tables.return_value = etl_tables or [
        "clinical_patient",
        "mutations_extended",
    ]
    cols = cols or ["PATIENT_ID", "AGENT", "START_DATE"]
    count_df = pd.DataFrame({"cnt": [row_count]})
    desc_df = pd.DataFrame({"column_name": cols})
    db.execute.side_effect = [count_df, desc_df]
    return db


def test_creates_view_and_returns_summary():
    db = _make_db(row_count=100, cols=["PATIENT_ID", "AGENT"])
    result = create_derived_view(
        db, "derived_regimens", "SELECT PATIENT_ID, AGENT FROM timeline_treatment"
    )
    db.create_view.assert_called_once_with(
        "derived_regimens", "SELECT PATIENT_ID, AGENT FROM timeline_treatment"
    )
    assert "Created view 'derived_regimens'" in result
    assert "100" in result
    assert "PATIENT_ID" in result
    assert "AGENT" in result


def test_adds_derived_prefix_if_missing():
    db = _make_db()
    result = create_derived_view(
        db, "regimens", "SELECT PATIENT_ID FROM timeline_treatment"
    )
    db.create_view.assert_called_once_with(
        "derived_regimens", "SELECT PATIENT_ID FROM timeline_treatment"
    )
    assert "derived_regimens" in result


def test_accepts_existing_prefix():
    db = _make_db()
    result = create_derived_view(
        db, "derived_regimens", "SELECT PATIENT_ID FROM timeline_treatment"
    )
    db.create_view.assert_called_once_with(
        "derived_regimens", "SELECT PATIENT_ID FROM timeline_treatment"
    )
    assert not result.startswith("ERROR")


def test_rejects_non_select():
    for bad_sql in [
        "UPDATE foo SET x = 1",
        "DROP TABLE foo",
        "INSERT INTO foo VALUES (1)",
        "DELETE FROM foo",
    ]:
        db = MagicMock()
        db.list_tables.return_value = []
        result = create_derived_view(db, "bad", bad_sql)
        assert result.startswith("ERROR"), f"Expected ERROR for: {bad_sql!r}"
        db.create_view.assert_not_called()


def test_accepts_cte():
    db = _make_db()
    cte_sql = (
        "WITH ranked AS (SELECT PATIENT_ID FROM timeline_treatment) "
        "SELECT * FROM ranked"
    )
    result = create_derived_view(db, "cte_view", cte_sql)
    assert not result.startswith("ERROR")
    db.create_view.assert_called_once()


def test_rejects_etl_shadowing():
    db = MagicMock()
    db.list_tables.return_value = ["clinical_patient", "mutations_extended"]
    result = create_derived_view(
        db, "clinical_patient", "SELECT * FROM clinical_patient"
    )
    assert result.startswith("ERROR")
    assert "shadow" in result
    db.create_view.assert_not_called()


def test_handles_sql_error():
    db = MagicMock()
    db.list_tables.return_value = []
    db.create_view.side_effect = Exception("Catalog error: column not found")
    result = create_derived_view(db, "bad_view", "SELECT nonexistent FROM foo")
    assert result.startswith("ERROR")
    assert "Exception" in result or "Catalog" in result


def test_col_truncation_at_ten():
    """More than 10 columns shows truncation message."""
    cols = [f"COL_{i}" for i in range(15)]
    db = _make_db(cols=cols)
    result = create_derived_view(db, "wide", "SELECT * FROM foo")
    assert "15 total" in result
