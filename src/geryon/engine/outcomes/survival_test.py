"""Tests for overall survival outcome handler dtype coercion."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from geryon.engine.outcomes.survival import OverallSurvivalHandler
from geryon.lang.outcomes import OverallSurvival


def _make_handler_with_mock_db(df: pd.DataFrame):
    """Return (handler, mock_db) with mock_db.execute returning df."""
    mock_db = MagicMock()
    mock_db.execute.return_value = df
    handler = OverallSurvivalHandler()
    return handler, mock_db


def test_extract_data_coerces_string_columns_to_numeric():
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2"],
            "os_months": ["12.5", "8.3"],
            "event": ["1", "0"],
            "dx_days": [-365.0, -365.0],
        }
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1", "P2"], OverallSurvival(), mock_db)

    assert len(result) == 2
    assert np.issubdtype(result["time"].dtype, np.number)
    assert np.issubdtype(result["event"].dtype, np.number)
    assert list(result["event"]) == [1, 0]


def test_extract_data_parses_cbioportal_status_format():
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2"],
            "os_months": [12.5, 8.3],
            "event": ["1:DECEASED", "0:LIVING"],
            "dx_days": [-365.0, -365.0],
        }
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1", "P2"], OverallSurvival(), mock_db)

    assert list(result["event"]) == [1, 0]
    assert np.issubdtype(result["event"].dtype, np.number)


def test_extract_data_drops_unparseable_rows():
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2", "P3"],
            "os_months": [12.5, "N/A", 8.0],
            "event": ["1:DECEASED", "0:LIVING", "UNKNOWN"],
            "dx_days": [-365.0, -365.0, -365.0],
        }
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1", "P2", "P3"], OverallSurvival(), mock_db)

    assert len(result) == 1
    assert list(result["PATIENT_ID"]) == ["P1"]


def test_diagnosis_anchor_computes_time_and_entry():
    # dx_days=-730 means diagnosis was 730 days (≈24 months) before sequencing
    # os_months=12 means patient survived 12 months from sequencing
    # → time ≈ 12 + 24 = 36 months from diagnosis
    # → entry_time ≈ 24 months (time from dx to sequencing)
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "os_months": [12.0],
            "event": [1],
            "dx_days": [-730.0],
        }
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1"], OverallSurvival(), mock_db)

    assert len(result) == 1
    assert abs(result["entry_time"].iloc[0] - 730 / 30.44) < 0.1
    assert abs(result["time"].iloc[0] - (12.0 + 730 / 30.44)) < 0.1


def test_no_dx_record_dropped():
    # Patient without a timeline_diagnosis record has no diagnosis anchor
    # and is excluded from the analysis.
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "os_months": [18.0],
            "event": [0],
            "dx_days": [None],
        }
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1"], OverallSurvival(), mock_db)

    assert len(result) == 0


def test_zero_os_months_dropped():
    # os_months=0 with dx_days=-365 → time ≈ entry_time; row excluded by filter
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "os_months": [0.0],
            "event": [1],
            "dx_days": [-365.0],
        }
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1"], OverallSurvival(), mock_db)

    assert len(result) == 0
