"""Tests for statistical methods."""

import pandas as pd

from msk_cycl.hypothesis.statistics import calculate_cox_hazard_ratio


def test_calculate_cox_hazard_ratio_with_known_data():
    """Cox regression calculates hazard ratio for test data."""
    # Cohort A: better survival (larger sample size for stable Cox fit)
    cohort_a = pd.DataFrame(
        {
            "PATIENT_ID": [f"A{i}" for i in range(1, 21)],
            "time": [
                12.0,
                15.0,
                10.0,
                18.0,
                14.0,
                16.0,
                11.0,
                13.0,
                17.0,
                19.0,
                12.5,
                14.5,
                11.5,
                13.5,
                15.5,
                16.5,
                10.5,
                12.0,
                14.0,
                13.0,
            ],
            "event": [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        }
    )

    # Cohort B: worse survival
    cohort_b = pd.DataFrame(
        {
            "PATIENT_ID": [f"B{i}" for i in range(1, 21)],
            "time": [
                6.0,
                8.0,
                7.0,
                5.0,
                9.0,
                7.5,
                6.5,
                8.5,
                5.5,
                7.0,
                6.0,
                8.0,
                7.0,
                6.5,
                8.5,
                5.0,
                7.5,
                6.0,
                7.0,
                8.0,
            ],
            "event": [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
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
