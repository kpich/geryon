"""Tests for hypothesis result models."""

import pandas as pd  # type: ignore

from msk_cycl.lang.results import ComparisonResult


def test_comparison_result_default_success_is_true():
    """ComparisonResult defaults to success=True."""
    result = ComparisonResult(
        cohort_a_ids=["P001"],
        cohort_b_ids=["P002"],
        cohort_a_size=1,
        cohort_b_size=1,
        cohort_a_data=pd.DataFrame(),
        cohort_b_data=pd.DataFrame(),
    )
    assert result.success is True
    assert result.error_message is None


def test_comparison_result_with_explicit_failure():
    """ComparisonResult can be created with success=False and error_message."""
    result = ComparisonResult(
        cohort_a_ids=[],
        cohort_b_ids=[],
        cohort_a_size=0,
        cohort_b_size=0,
        cohort_a_data=pd.DataFrame(),
        cohort_b_data=pd.DataFrame(),
        success=False,
        error_message="Database connection failed",
    )
    assert result.success is False
    assert result.error_message == "Database connection failed"


def test_comparison_result_serialization_preserves_success_fields():
    """Serialization round-trip preserves success and error_message."""
    result = ComparisonResult(
        cohort_a_ids=[],
        cohort_b_ids=[],
        cohort_a_size=0,
        cohort_b_size=0,
        cohort_a_data=pd.DataFrame(),
        cohort_b_data=pd.DataFrame(),
        success=False,
        error_message="Test error",
    )

    data = result.model_dump()
    assert data["success"] is False
    assert data["error_message"] == "Test error"


def test_comparison_result_with_stats_and_success():
    """ComparisonResult can have both statistics and success fields."""
    result = ComparisonResult(
        cohort_a_ids=["P001", "P002"],
        cohort_b_ids=["P003", "P004"],
        cohort_a_size=2,
        cohort_b_size=2,
        cohort_a_data=pd.DataFrame({"OS_MONTHS": [12.0, 15.0]}),
        cohort_b_data=pd.DataFrame({"OS_MONTHS": [8.0, 10.0]}),
        hazard_ratio=0.75,
        p_value=0.05,
        success=True,
    )
    assert result.success is True
    assert result.hazard_ratio == 0.75
    assert result.p_value == 0.05
