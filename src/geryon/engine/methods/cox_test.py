"""Tests for statistical methods."""

import pandas as pd
import pytest

from geryon.engine.methods.cox import CoxHazardRatioMethod


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::lifelines.utils.ConvergenceWarning")
def test_calculate_cox_hazard_ratio_with_known_data():
    """Cox regression calculates hazard ratio for test data."""
    # Cohort A: better survival
    cohort_a = pd.DataFrame(
        {
            "PATIENT_ID": ["A1", "A2", "A3", "A4", "A5"],
            "time": [12.0, 15.0, 10.0, 18.0, 14.0],
            "event": [1, 0, 1, 0, 1],
        }
    )

    # Cohort B: worse survival
    cohort_b = pd.DataFrame(
        {
            "PATIENT_ID": ["B1", "B2", "B3", "B4", "B5"],
            "time": [6.0, 8.0, 7.0, 5.0, 9.0],
            "event": [1, 1, 1, 1, 1],
        }
    )

    method = CoxHazardRatioMethod()
    result = method.calculate(cohort_a, cohort_b)

    assert "hazard_ratio" in result
    assert "confidence_interval_lower" in result
    assert "confidence_interval_upper" in result
    assert "p_value" in result

    # Cohort A should have lower hazard (better survival)
    assert result["hazard_ratio"] is not None and result["hazard_ratio"] > 0
    assert result["p_value"] is not None
    assert 0 <= result["p_value"] <= 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::lifelines.utils.ConvergenceWarning")
def test_calculate_with_left_truncation():
    """Cox regression handles left-truncated data via entry_col."""
    cohort_a = pd.DataFrame(
        {
            "PATIENT_ID": ["A1", "A2", "A3", "A4", "A5"],
            "time": [24.0, 30.0, 20.0, 36.0, 28.0],
            "event": [1, 0, 1, 0, 1],
            "entry_time": [2.0, 3.0, 1.0, 4.0, 2.0],
        }
    )
    cohort_b = pd.DataFrame(
        {
            "PATIENT_ID": ["B1", "B2", "B3", "B4", "B5"],
            "time": [18.0, 22.0, 19.0, 17.0, 21.0],
            "event": [1, 1, 1, 1, 1],
            "entry_time": [1.0, 2.0, 1.5, 1.0, 2.0],
        }
    )

    method = CoxHazardRatioMethod()
    result = method.calculate(cohort_a, cohort_b)

    assert "hazard_ratio" in result
    assert "confidence_interval_lower" in result
    assert "confidence_interval_upper" in result
    assert "p_value" in result
    assert result["hazard_ratio"] is not None and result["hazard_ratio"] > 0


def test_cox_rejects_cbioportal_status_strings():
    """Cox regression fails on cBioPortal '1:DECEASED'/'0:LIVING' strings."""
    cohort_a = pd.DataFrame(
        {
            "PATIENT_ID": ["A1", "A2"],
            "time": [12.0, 15.0],
            "event": ["1:DECEASED", "0:LIVING"],
        }
    )
    cohort_b = pd.DataFrame(
        {
            "PATIENT_ID": ["B1", "B2"],
            "time": [6.0, 8.0],
            "event": ["1:DECEASED", "1:DECEASED"],
        }
    )

    method = CoxHazardRatioMethod()
    with pytest.raises(TypeError, match="Wrong dtype"):
        method.calculate(cohort_a, cohort_b)
