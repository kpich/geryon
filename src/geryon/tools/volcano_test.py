"""Unit tests for volcano scan tool."""

from unittest.mock import Mock

import pandas as pd

from geryon.lang.outcomes import OverallSurvival, TimeToNextTreatment
from geryon.lang.results import ComparisonResult
from geryon.tools.volcano import _parse_outcome, scan_groupby


def _make_result(
    cohort_a_size: int = 15,
    cohort_b_size: int = 85,
    hazard_ratio: float = 1.5,
    p_value: float = 0.05,
    success: bool = True,
) -> ComparisonResult:
    return ComparisonResult(
        cohort_a_size=cohort_a_size,
        cohort_b_size=cohort_b_size,
        hazard_ratio=hazard_ratio,
        p_value=p_value,
        success=success,
        error_message=None if success else "comparison failed",
    )


def _make_batch_df(groups: dict[str, list[str]]) -> pd.DataFrame:
    """Build the batch-query DataFrame from a {group_value: [patient_ids]} dict."""
    rows = [
        {"grp": grp, "PATIENT_ID": pid} for grp, pids in groups.items() for pid in pids
    ]
    return pd.DataFrame(rows)


def _all_ids_df(groups: dict[str, list[str]]) -> pd.DataFrame:
    all_pids = [pid for pids in groups.values() for pid in pids]
    return pd.DataFrame({"PATIENT_ID": all_pids})


# ---------------------------------------------------------------------------
# test_scan_groupby_returns_top_n
# ---------------------------------------------------------------------------


def test_scan_groupby_returns_top_n():
    """3 groups all succeed; top_n=2 returns at most 2 data rows."""
    groups = {
        "A": [f"PA{i}" for i in range(15)],
        "B": [f"PB{i}" for i in range(15)],
        "C": [f"PC{i}" for i in range(15)],
    }
    mock_db = Mock()
    mock_db.execute.side_effect = [_make_batch_df(groups), _all_ids_df(groups)]

    mock_executor = Mock()
    mock_executor.compare_ids.side_effect = [
        _make_result(p_value=0.001),
        _make_result(p_value=0.01),
        _make_result(p_value=0.05),
    ]

    result = scan_groupby(
        mock_db,
        mock_executor,
        group_table="clinical_patient",
        group_column="CANCER_TYPE",
        outcome_spec='{"outcome_type": "overall_survival"}',
        top_n=2,
    )

    data_rows = [
        line
        for line in result.split("\n")
        if line.startswith("|") and "---" not in line and "Group" not in line
    ]
    assert len(data_rows) <= 2
    assert "|" in result


# ---------------------------------------------------------------------------
# test_scan_groupby_filters_small_groups
# ---------------------------------------------------------------------------


def test_scan_groupby_filters_small_groups():
    """Groups with fewer than min_group_size patients are excluded before comparison."""
    groups = {
        "BIG": [f"PB{i}" for i in range(20)],
        "TINY": ["PT0", "PT1"],  # only 2 — below min_group_size=10
    }
    mock_db = Mock()
    mock_db.execute.side_effect = [_make_batch_df(groups), _all_ids_df(groups)]

    mock_executor = Mock()
    mock_executor.compare_ids.return_value = _make_result(p_value=0.01)

    result = scan_groupby(
        mock_db,
        mock_executor,
        group_table="clinical_patient",
        group_column="CANCER_TYPE",
        outcome_spec='{"outcome_type": "overall_survival"}',
        min_group_size=10,
    )

    # compare_ids should only be called once (for BIG, not TINY)
    assert mock_executor.compare_ids.call_count == 1
    assert "TINY" not in result


# ---------------------------------------------------------------------------
# test_scan_groupby_handles_failed_comparison
# ---------------------------------------------------------------------------


def test_scan_groupby_handles_failed_comparison():
    """Groups whose comparison returns success=False are excluded from the table."""
    groups = {
        "OK": [f"PO{i}" for i in range(15)],
        "FAIL": [f"PF{i}" for i in range(15)],
    }
    mock_db = Mock()
    mock_db.execute.side_effect = [_make_batch_df(groups), _all_ids_df(groups)]

    mock_executor = Mock()
    # Return order matches iteration order (most-frequent first — tied, so dict order)
    mock_executor.compare_ids.side_effect = [
        _make_result(p_value=0.02),  # OK group succeeds
        _make_result(success=False),  # FAIL group fails
    ]

    result = scan_groupby(
        mock_db,
        mock_executor,
        group_table="clinical_patient",
        group_column="CANCER_TYPE",
        outcome_spec='{"outcome_type": "overall_survival"}',
    )

    data_rows = [
        line
        for line in result.split("\n")
        if line.startswith("|") and "---" not in line and "Group" not in line
    ]
    assert len(data_rows) == 1


# ---------------------------------------------------------------------------
# test_scan_groupby_applies_fdr_correction
# ---------------------------------------------------------------------------


def test_scan_groupby_applies_fdr_correction():
    """q_value column is present in output after BH FDR correction."""
    groups = {
        "A": [f"PA{i}" for i in range(15)],
        "B": [f"PB{i}" for i in range(15)],
    }
    mock_db = Mock()
    mock_db.execute.side_effect = [_make_batch_df(groups), _all_ids_df(groups)]

    mock_executor = Mock()
    mock_executor.compare_ids.side_effect = [
        _make_result(p_value=0.01),
        _make_result(p_value=0.04),
    ]

    result = scan_groupby(
        mock_db,
        mock_executor,
        group_table="clinical_patient",
        group_column="CANCER_TYPE",
        outcome_spec='{"outcome_type": "overall_survival"}',
    )

    assert "q_value" in result


# ---------------------------------------------------------------------------
# test_parse_outcome_overall_survival
# ---------------------------------------------------------------------------


def test_parse_outcome_overall_survival():
    """JSON with outcome_type='overall_survival' parses to OverallSurvival."""
    outcome = _parse_outcome('{"outcome_type": "overall_survival"}')
    assert isinstance(outcome, OverallSurvival)


# ---------------------------------------------------------------------------
# test_parse_outcome_ttnt_with_agent
# ---------------------------------------------------------------------------


def test_parse_outcome_ttnt_with_agent():
    """JSON with outcome_type='time_to_next_treatment', agent field parses correctly."""
    outcome = _parse_outcome(
        '{"outcome_type": "time_to_next_treatment", "agent": "Sotorasib"}'
    )
    assert isinstance(outcome, TimeToNextTreatment)
    assert outcome.agent == "Sotorasib"
