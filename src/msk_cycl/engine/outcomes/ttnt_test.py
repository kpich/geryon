"""Tests for TTNT outcome handler."""

from unittest.mock import MagicMock

import pandas as pd

from msk_cycl.engine.outcomes.ttnt import DAYS_PER_MONTH, TimeToNextTreatmentHandler
from msk_cycl.lang.outcomes import TimeToNextTreatment


def _make_handler():
    return TimeToNextTreatmentHandler()


def test_empty_cohort_returns_empty_dataframe():
    handler = _make_handler()
    mock_db = MagicMock()
    outcome = TimeToNextTreatment(agent="Pembrolizumab")

    result = handler.extract_data([], outcome, mock_db)

    assert list(result.columns) == ["PATIENT_ID", "time", "event"]
    assert len(result) == 0


def test_patient_with_next_treatment_is_event():
    handler = _make_handler()
    mock_db = MagicMock()

    # First call: main SQL returning index + next treatment
    index_start = 100
    next_start = 300  # 200 days after index, > 90 gap
    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "index_start": [index_start],
            "next_start": [float(next_start)],
        }
    )

    outcome = TimeToNextTreatment(agent="Pembrolizumab", gap_days=90)
    result = handler.extract_data(["P1"], outcome, mock_db)

    assert len(result) == 1
    assert result.iloc[0]["event"] == 1
    expected_time = (next_start - index_start) / DAYS_PER_MONTH
    assert abs(result.iloc[0]["time"] - expected_time) < 0.01


def test_patient_without_next_treatment_is_censored():
    handler = _make_handler()
    mock_db = MagicMock()

    index_start = 100
    os_months = 24.0

    # First call: main SQL (next_start is null)
    # Second call: OS data for censored patients
    mock_db.execute.side_effect = [
        pd.DataFrame(
            {
                "PATIENT_ID": ["P1"],
                "index_start": [index_start],
                "next_start": [None],
            }
        ),
        pd.DataFrame({"PATIENT_ID": ["P1"], "OS_MONTHS": [os_months]}),
    ]

    outcome = TimeToNextTreatment(agent="Pembrolizumab", gap_days=90)
    result = handler.extract_data(["P1"], outcome, mock_db)

    assert len(result) == 1
    assert result.iloc[0]["event"] == 0
    expected_time = os_months - index_start / DAYS_PER_MONTH
    assert abs(result.iloc[0]["time"] - expected_time) < 0.01


def test_negative_time_patients_are_excluded():
    handler = _make_handler()
    mock_db = MagicMock()

    # Patient with OS_MONTHS < index_start / DAYS_PER_MONTH → negative time
    mock_db.execute.side_effect = [
        pd.DataFrame(
            {
                "PATIENT_ID": ["P1"],
                "index_start": [1000],  # ~33 months
                "next_start": [None],
            }
        ),
        pd.DataFrame({"PATIENT_ID": ["P1"], "OS_MONTHS": [10.0]}),  # < 33
    ]

    outcome = TimeToNextTreatment(agent="Pembrolizumab", gap_days=90)
    result = handler.extract_data(["P1"], outcome, mock_db)

    assert len(result) == 0


def test_mixed_event_and_censored_patients():
    handler = _make_handler()
    mock_db = MagicMock()

    mock_db.execute.side_effect = [
        pd.DataFrame(
            {
                "PATIENT_ID": ["P1", "P2"],
                "index_start": [100, 200],
                "next_start": [300.0, None],
            }
        ),
        pd.DataFrame({"PATIENT_ID": ["P2"], "OS_MONTHS": [36.0]}),
    ]

    outcome = TimeToNextTreatment(agent="Pembrolizumab", gap_days=90)
    result = handler.extract_data(["P1", "P2"], outcome, mock_db)

    assert len(result) == 2
    p1 = result[result["PATIENT_ID"] == "P1"].iloc[0]
    p2 = result[result["PATIENT_ID"] == "P2"].iloc[0]
    assert p1["event"] == 1
    assert p2["event"] == 0


def test_no_patients_with_agent_returns_empty():
    handler = _make_handler()
    mock_db = MagicMock()
    mock_db.execute.return_value = pd.DataFrame(
        columns=["PATIENT_ID", "index_start", "next_start"]
    )

    outcome = TimeToNextTreatment(agent="NonexistentDrug", gap_days=90)
    result = handler.extract_data(["P1"], outcome, mock_db)

    assert len(result) == 0
