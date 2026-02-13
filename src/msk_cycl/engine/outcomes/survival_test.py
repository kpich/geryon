"""Tests for overall survival outcome handler dtype coercion."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from msk_cycl.engine.outcomes.survival import OverallSurvivalHandler
from msk_cycl.lang.outcomes import OverallSurvival


def _make_handler_with_mock_db(df: pd.DataFrame):
    """Return (handler, mock_db) with mock_db.execute returning df."""
    mock_db = MagicMock()
    mock_db.execute.return_value = df
    handler = OverallSurvivalHandler()
    return handler, mock_db


def test_extract_data_coerces_string_columns_to_numeric():
    df = pd.DataFrame(
        {"PATIENT_ID": ["P1", "P2"], "time": ["12.5", "8.3"], "event": ["1", "0"]}
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1", "P2"], OverallSurvival(), mock_db)

    assert np.issubdtype(result["time"].dtype, np.number)
    assert np.issubdtype(result["event"].dtype, np.number)
    assert list(result["time"]) == [12.5, 8.3]
    assert list(result["event"]) == [1, 0]


def test_extract_data_parses_cbioportal_status_format():
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2"],
            "time": [12.5, 8.3],
            "event": ["1:DECEASED", "0:LIVING"],
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
            "time": [12.5, "N/A", 8.0],
            "event": ["1:DECEASED", "0:LIVING", "UNKNOWN"],
        }
    )
    handler, mock_db = _make_handler_with_mock_db(df)

    result = handler.extract_data(["P1", "P2", "P3"], OverallSurvival(), mock_db)

    assert len(result) == 1
    assert list(result["PATIENT_ID"]) == ["P1"]
