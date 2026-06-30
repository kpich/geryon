"""Tests for progression-from-treatment outcome handler."""

from unittest.mock import MagicMock

import pandas as pd

from geryon.legacy.engine.outcomes.progression_from_treatment import (
    DAYS_PER_MONTH,
    ProgressionFromTreatmentHandler,
)
from geryon.legacy.lang.outcomes import ProgressionFromTreatment


def _make_handler():
    return ProgressionFromTreatmentHandler()


def _outcome():
    return ProgressionFromTreatment(agent="Pembrolizumab")


def test_empty_cohort_returns_empty_dataframe():
    handler = _make_handler()
    mock_db = MagicMock()
    result = handler.extract_data([], _outcome(), mock_db)
    assert list(result.columns) == ["PATIENT_ID", "time", "event", "entry_time"]
    assert len(result) == 0


def test_entry_time_left_truncation():
    """Treatment before sequencing enters the risk set at sequencing, not at tx."""
    handler = _make_handler()
    mock_db = MagicMock()

    # tx_start = -304.4 days (10 months pre-sequencing); progression after tx.
    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "tx_start": [-304.4],
            "OS_MONTHS": [24.0],
            "OS_STATUS": ["1:DECEASED"],
            "prog_date": [float("nan")],
        }
    )

    result = handler.extract_data(["P1"], _outcome(), mock_db)

    assert len(result) == 1
    assert abs(result.iloc[0]["entry_time"] - 10.0) < 0.01


def test_progression_before_death():
    """When radiological progression occurs before death, PFS uses prog event."""
    handler = _make_handler()
    mock_db = MagicMock()

    tx_start = 0
    prog_date = 60  # day 60 → 60/30.44 months from tx_start
    os_months = 24.0  # death much later

    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "tx_start": [tx_start],
            "OS_MONTHS": [os_months],
            "OS_STATUS": ["1:DECEASED"],
            "prog_date": [prog_date],
        }
    )

    result = handler.extract_data(["P1"], _outcome(), mock_db)

    expected_time = prog_date / DAYS_PER_MONTH
    assert len(result) == 1
    assert abs(result.iloc[0]["time"] - expected_time) < 0.01
    assert result.iloc[0]["event"] == 1


def test_death_before_progression():
    """When death occurs before progression, PFS uses OS event."""
    handler = _make_handler()
    mock_db = MagicMock()

    tx_start = 0
    prog_date = 900  # day 900 — after death
    os_months = 12.0

    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "tx_start": [tx_start],
            "OS_MONTHS": [os_months],
            "OS_STATUS": ["1:DECEASED"],
            "prog_date": [prog_date],
        }
    )

    result = handler.extract_data(["P1"], _outcome(), mock_db)

    assert len(result) == 1
    assert abs(result.iloc[0]["time"] - os_months) < 0.01
    assert result.iloc[0]["event"] == 1


def test_no_progression_alive():
    """Patient alive with no progression is censored at OS time."""
    handler = _make_handler()
    mock_db = MagicMock()

    tx_start = 0
    os_months = 18.0

    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "tx_start": [tx_start],
            "OS_MONTHS": [os_months],
            "OS_STATUS": ["0:LIVING"],
            "prog_date": [float("nan")],
        }
    )

    result = handler.extract_data(["P1"], _outcome(), mock_db)

    assert len(result) == 1
    assert abs(result.iloc[0]["time"] - os_months) < 0.01
    assert result.iloc[0]["event"] == 0


def test_negative_time_excluded():
    """Patients with time <= 0 are dropped."""
    handler = _make_handler()
    mock_db = MagicMock()

    # tx_start at day 1000 (~33 months), OS at 10 months → negative os_time
    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "tx_start": [1000],
            "OS_MONTHS": [10.0],
            "OS_STATUS": ["1:DECEASED"],
            "prog_date": [float("nan")],
        }
    )

    result = handler.extract_data(["P1"], _outcome(), mock_db)

    assert len(result) == 0
