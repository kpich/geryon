"""Tests for hypothesis storage."""

import json
from pathlib import Path

import pandas as pd  # type: ignore
import pytest

from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.storage import SUPPORTED_VERSIONS, HypothesisStore
from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CohortFilter, CompareCohorts, CyclHyp, SelectCohort
from msk_cycl.llm.generator import HypothesisProposal
from msk_cycl.llm.narrator import HypothesisNarrative


def _make_hypothesis(
    hypothesis_id: str = "test-id",
    session_id: str = "test-session",
    cohort_a_data: pd.DataFrame | None = None,
    cohort_b_data: pd.DataFrame | None = None,
) -> LabeledHypothesis:
    """Create a test hypothesis with given DataFrames."""
    if cohort_a_data is None:
        cohort_a_data = pd.DataFrame({"OS_MONTHS": [12.0], "OS_STATUS": [1]})
    if cohort_b_data is None:
        cohort_b_data = pd.DataFrame({"OS_MONTHS": [8.0], "OS_STATUS": [1]})

    spec = CyclHyp(
        query=CompareCohorts(
            cohort_a=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="TREATMENT",
                        operator="==",
                        value="Drug A",
                    )
                ]
            ),
            cohort_b=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="TREATMENT",
                        operator="==",
                        value="Placebo",
                    )
                ]
            ),
            outcome=OverallSurvival(),
            method=ComparisonMethod.HAZARD_RATIO_COX,
        )
    )

    return LabeledHypothesis(
        hypothesis_id=hypothesis_id,
        session_id=session_id,
        proposal=HypothesisProposal(
            cohort_a_description="Drug A patients",
            cohort_b_description="Placebo patients",
            outcome_description="Overall survival",
            rationale="Test rationale",
            cycl_spec=spec,
        ),
        spec=spec,
        result=ComparisonResult(
            cohort_a_ids=["P001"],
            cohort_b_ids=["P002"],
            cohort_a_size=1,
            cohort_b_size=1,
            cohort_a_data=cohort_a_data,
            cohort_b_data=cohort_b_data,
        ),
        execution_time_seconds=0.5,
        narrative=HypothesisNarrative(
            summary="Test summary",
            findings="Test findings",
            limitations=["Test limitation"],
            clinical_relevance="Test relevance",
        ),
        llm_model="test-model",
    )


def test_storage_round_trip_preserves_dtypes(tmp_path: Path):
    """Save and load preserves DataFrame dtypes."""
    df_a = pd.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "age": [65, 70],
            "survival_months": [12.5, 8.3],
        }
    )
    df_b = pd.DataFrame(
        {
            "patient_id": ["P003", "P004"],
            "age": [55, 60],
            "survival_months": [15.2, 9.1],
        }
    )

    hypothesis = _make_hypothesis(
        cohort_a_data=df_a,
        cohort_b_data=df_b,
    )

    store = HypothesisStore(tmp_path)
    store.save(hypothesis)

    loaded = store.load_session("test-session")
    assert len(loaded) == 1

    loaded_df_a = loaded[0].result.cohort_a_data
    loaded_df_b = loaded[0].result.cohort_b_data

    assert loaded_df_a["patient_id"].dtype == df_a["patient_id"].dtype
    assert loaded_df_a["age"].dtype == df_a["age"].dtype
    assert loaded_df_a["survival_months"].dtype == df_a["survival_months"].dtype

    pd.testing.assert_frame_equal(loaded_df_a, df_a)
    pd.testing.assert_frame_equal(loaded_df_b, df_b)


def test_storage_loads_old_list_format(tmp_path: Path):
    """Loading old format (list) still works for backwards compatibility."""
    store = HypothesisStore(tmp_path)

    hypothesis = _make_hypothesis()
    store.save(hypothesis)

    session_file = tmp_path / "test-session.jsonl"
    lines = session_file.read_text().splitlines()

    modified_lines = []
    for line in lines:
        data = json.loads(line)
        if data.get("record_type") == "hypothesis":
            result = data["data"]["result"]
            if "cohort_a_data" in result and isinstance(result["cohort_a_data"], dict):
                result["cohort_a_data"] = result["cohort_a_data"]["records"]
            if "cohort_b_data" in result and isinstance(result["cohort_b_data"], dict):
                result["cohort_b_data"] = result["cohort_b_data"]["records"]
        modified_lines.append(json.dumps(data, default=str))

    session_file.write_text("\n".join(modified_lines) + "\n")

    loaded = store.load_session("test-session")
    assert len(loaded) == 1
    assert isinstance(loaded[0].result.cohort_a_data, pd.DataFrame)
    assert isinstance(loaded[0].result.cohort_b_data, pd.DataFrame)


def test_storage_version_check_accepts_version_1(tmp_path: Path):
    """Loading version 1 succeeds."""
    assert 1 in SUPPORTED_VERSIONS

    hypothesis = _make_hypothesis()
    assert hypothesis.spec.version == 1

    store = HypothesisStore(tmp_path)
    store.save(hypothesis)

    loaded = store.load_session("test-session")
    assert len(loaded) == 1


def test_storage_version_check_rejects_unsupported_version(tmp_path: Path):
    """Loading unsupported version raises ValueError."""
    store = HypothesisStore(tmp_path)

    hypothesis = _make_hypothesis()
    store.save(hypothesis)

    session_file = tmp_path / "test-session.jsonl"
    lines = session_file.read_text().splitlines()

    modified_lines = []
    for line in lines:
        data = json.loads(line)
        if data.get("record_type") == "hypothesis":
            data["data"]["spec"]["version"] = 99
        modified_lines.append(json.dumps(data, default=str))

    session_file.write_text("\n".join(modified_lines) + "\n")

    with pytest.raises(ValueError) as exc_info:
        store.load_session("test-session")

    assert "Unsupported CyclHyp version: 99" in str(exc_info.value)


def test_storage_empty_session_returns_empty_list(tmp_path: Path):
    """Loading non-existent session returns empty list."""
    store = HypothesisStore(tmp_path)
    loaded = store.load_session("nonexistent")
    assert loaded == []
