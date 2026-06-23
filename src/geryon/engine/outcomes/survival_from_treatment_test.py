"""Tests for survival from treatment outcome handler."""

from unittest.mock import MagicMock

import pandas as pd

from geryon.engine.outcomes.survival_from_treatment import (
    DAYS_PER_MONTH,
    SurvivalFromTreatmentHandler,
)
from geryon.lang.outcomes import SurvivalFromTreatment


def _make_handler():
    return SurvivalFromTreatmentHandler()


def test_empty_cohort_returns_empty_dataframe():
    handler = _make_handler()
    mock_db = MagicMock()
    outcome = SurvivalFromTreatment(agent="Carboplatin")

    result = handler.extract_data([], outcome, mock_db)

    assert list(result.columns) == ["PATIENT_ID", "time", "event", "entry_time"]
    assert len(result) == 0


def test_entry_time_left_truncation():
    handler = _make_handler()
    mock_db = MagicMock()

    # P1 treated 304.4 days (10 months) BEFORE sequencing -> immortal lead time;
    # P2 treated 30.44 days (1 month) AFTER sequencing -> enters at treatment.
    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2"],
            "tx_start": [-304.4, 30.44],
            "OS_MONTHS": [12.0, 12.0],
            "OS_STATUS": ["1:DECEASED", "1:DECEASED"],
        }
    )

    outcome = SurvivalFromTreatment(agent="Carboplatin")
    result = handler.extract_data(["P1", "P2"], outcome, mock_db)

    entry = dict(zip(result["PATIENT_ID"], result["entry_time"], strict=False))
    assert abs(entry["P1"] - 10.0) < 0.01  # enters risk set at sequencing
    assert entry["P2"] == 0.0  # post-sequencing start enters at treatment
    # observed follow-up (time - entry_time) stays positive for both
    assert len(result) == 2


def test_adjusted_time_computation():
    handler = _make_handler()
    mock_db = MagicMock()

    tx_start = 300  # days
    os_months = 24.0

    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "tx_start": [tx_start],
            "OS_MONTHS": [os_months],
            "OS_STATUS": ["1:DECEASED"],
        }
    )

    outcome = SurvivalFromTreatment(agent="Carboplatin")
    result = handler.extract_data(["P1"], outcome, mock_db)

    expected_time = os_months - tx_start / DAYS_PER_MONTH
    assert len(result) == 1
    assert abs(result.iloc[0]["time"] - expected_time) < 0.01
    assert result.iloc[0]["event"] == 1


def test_negative_time_excluded():
    handler = _make_handler()
    mock_db = MagicMock()

    # tx_start at day 1000 (~33 months), OS at 10 months → negative
    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "tx_start": [1000],
            "OS_MONTHS": [10.0],
            "OS_STATUS": ["1:DECEASED"],
        }
    )

    outcome = SurvivalFromTreatment(agent="Carboplatin")
    result = handler.extract_data(["P1"], outcome, mock_db)

    assert len(result) == 0


def test_cbioportal_status_parsing():
    handler = _make_handler()
    mock_db = MagicMock()

    mock_db.execute.return_value = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2"],
            "tx_start": [100, 100],
            "OS_MONTHS": [24.0, 30.0],
            "OS_STATUS": ["1:DECEASED", "0:LIVING"],
        }
    )

    outcome = SurvivalFromTreatment(agent="Carboplatin")
    result = handler.extract_data(["P1", "P2"], outcome, mock_db)

    assert len(result) == 2
    events = dict(zip(result["PATIENT_ID"], result["event"], strict=False))
    assert events["P1"] == 1
    assert events["P2"] == 0


def test_missing_treatment_excluded():
    handler = _make_handler()
    mock_db = MagicMock()

    # SQL returns empty (no patients had the agent)
    mock_db.execute.return_value = pd.DataFrame(
        columns=["PATIENT_ID", "tx_start", "OS_MONTHS", "OS_STATUS"]
    )

    outcome = SurvivalFromTreatment(agent="NonexistentDrug")
    result = handler.extract_data(["P1"], outcome, mock_db)

    assert len(result) == 0
