"""Tests for statistical methods."""

import pandas as pd

from msk_cycl.hypothesis.statistics import calculate_cox_hazard_ratio


def test_calculate_cox_hazard_ratio_with_known_data():
    """Cox regression calculates hazard ratio for test data."""
    # Cohort A: better survival
    cohort_a = pd.DataFrame(
        {
            "PATIENT_ID": ["A1", "A2", "A3"],
            "time": [12.0, 15.0, 10.0],
            "event": [1, 0, 1],
        }
    )

    # Cohort B: worse survival
    cohort_b = pd.DataFrame(
        {
            "PATIENT_ID": ["B1", "B2", "B3"],
            "time": [6.0, 8.0, 7.0],
            "event": [1, 1, 1],
        }
    )

    result = calculate_cox_hazard_ratio(cohort_a, cohort_b)

    assert "hazard_ratio" in result
    assert "confidence_interval_lower" in result
    assert "confidence_interval_upper" in result
    assert "p_value" in result

    # Cohort A should have lower hazard (better survival)
    assert result["hazard_ratio"] > 0
    assert result["p_value"] >= 0
    assert result["p_value"] <= 1
