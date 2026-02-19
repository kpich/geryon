"""Tests for Wilcoxon rank-sum method."""

import pandas as pd

from msk_cycl.engine.methods.wilcoxon import WilcoxonRankSumMethod


def test_basic_two_group_comparison():
    cohort_a = pd.DataFrame({"PATIENT_ID": ["A1", "A2", "A3"], "value": [3, 5, 7]})
    cohort_b = pd.DataFrame({"PATIENT_ID": ["B1", "B2", "B3"], "value": [1, 2, 3]})

    method = WilcoxonRankSumMethod()
    result = method.calculate(cohort_a, cohort_b)

    assert result["median_a"] == 5.0
    assert result["median_b"] == 2.0
    assert result["u_statistic"] is not None
    assert result["p_value"] is not None
    assert 0 <= result["p_value"] <= 1


def test_identical_groups():
    data = pd.DataFrame({"PATIENT_ID": ["A1", "A2"], "value": [3, 3]})

    method = WilcoxonRankSumMethod()
    result = method.calculate(data.copy(), data.copy())

    assert result["median_a"] == 3.0
    assert result["median_b"] == 3.0
    assert result["p_value"] is not None


def test_empty_group_a_returns_none():
    cohort_a = pd.DataFrame({"PATIENT_ID": [], "value": []})
    cohort_b = pd.DataFrame({"PATIENT_ID": ["B1"], "value": [5]})

    method = WilcoxonRankSumMethod()
    result = method.calculate(cohort_a, cohort_b)

    assert result["median_a"] is None
    assert result["median_b"] is None
    assert result["u_statistic"] is None
    assert result["p_value"] is None


def test_empty_group_b_returns_none():
    cohort_a = pd.DataFrame({"PATIENT_ID": ["A1"], "value": [5]})
    cohort_b = pd.DataFrame({"PATIENT_ID": [], "value": []})

    method = WilcoxonRankSumMethod()
    result = method.calculate(cohort_a, cohort_b)

    assert result["median_a"] is None
    assert result["p_value"] is None
