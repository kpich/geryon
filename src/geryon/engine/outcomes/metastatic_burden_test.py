"""Tests for metastatic burden outcome handler."""

from unittest.mock import MagicMock

import pandas as pd

from geryon.engine.outcomes.metastatic_burden import MetastaticBurdenHandler
from geryon.lang.outcomes import MetastaticBurden


def _make_handler():
    return MetastaticBurdenHandler()


def test_empty_cohort_returns_empty_dataframe():
    handler = _make_handler()
    mock_db = MagicMock()
    outcome = MetastaticBurden()

    result = handler.extract_data([], outcome, mock_db)

    assert list(result.columns) == ["PATIENT_ID", "value"]
    assert len(result) == 0


def test_distinct_site_counting():
    handler = _make_handler()
    mock_db = MagicMock()

    # DB returns counts of distinct sites
    mock_db.execute.return_value = pd.DataFrame(
        {"PATIENT_ID": ["P1", "P2"], "value": [3, 1]}
    )

    outcome = MetastaticBurden(landmark_days=0, window_days=30)
    result = handler.extract_data(["P1", "P2"], outcome, mock_db)

    assert len(result) == 2
    values = dict(zip(result["PATIENT_ID"], result["value"], strict=False))
    assert values["P1"] == 3
    assert values["P2"] == 1


def test_no_tumor_sites_excluded_via_sql():
    handler = _make_handler()
    mock_db = MagicMock()

    # The SQL excludes "No Tumor Sites", so DB returns only real sites
    mock_db.execute.return_value = pd.DataFrame({"PATIENT_ID": ["P1"], "value": [2]})

    outcome = MetastaticBurden()
    result = handler.extract_data(["P1"], outcome, mock_db)

    # Verify the SQL contains the exclusion clause
    sql_called = mock_db.execute.call_args[0][0]
    assert "No Tumor Sites" in sql_called

    assert result.iloc[0]["value"] == 2


def test_patients_without_scans_get_zero():
    handler = _make_handler()
    mock_db = MagicMock()

    # Only P1 has scans; P2 does not appear in query results
    mock_db.execute.return_value = pd.DataFrame({"PATIENT_ID": ["P1"], "value": [2]})

    outcome = MetastaticBurden()
    result = handler.extract_data(["P1", "P2"], outcome, mock_db)

    assert len(result) == 2
    values = dict(zip(result["PATIENT_ID"], result["value"], strict=False))
    assert values["P1"] == 2
    assert values["P2"] == 0


def test_window_filtering_in_sql():
    handler = _make_handler()
    mock_db = MagicMock()
    mock_db.execute.return_value = pd.DataFrame(columns=["PATIENT_ID", "value"])

    outcome = MetastaticBurden(landmark_days=100, window_days=50)
    handler.extract_data(["P1"], outcome, mock_db)

    sql_called = mock_db.execute.call_args[0][0]
    assert "50" in sql_called  # lo = 100 - 50
    assert "150" in sql_called  # hi = 100 + 50
