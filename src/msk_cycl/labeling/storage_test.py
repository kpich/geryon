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


def test_storage_round_trip_excludes_bulk_data(tmp_path: Path):
    """Save and load drops cohort DataFrames (they are recomputable from spec)."""
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

    hypothesis = _make_hypothesis(cohort_a_data=df_a, cohort_b_data=df_b)

    store = HypothesisStore(tmp_path)
    store.save(hypothesis)

    loaded = store.load()
    assert len(loaded) == 1

    assert loaded[0].result.cohort_a_data is None
    assert loaded[0].result.cohort_b_data is None
    assert loaded[0].result.cohort_a_size == 1
    assert loaded[0].result.cohort_b_size == 1


def test_storage_jsonl_does_not_contain_bulk_fields(tmp_path: Path):
    """Serialized JSONL should not contain cohort_a_data or cohort_b_data."""
    hypothesis = _make_hypothesis()

    store = HypothesisStore(tmp_path)
    store.save(hypothesis)

    session_file = tmp_path / "hypotheses.jsonl"
    lines = session_file.read_text().splitlines()

    for line in lines:
        data = json.loads(line)
        if data.get("record_type") == "hypothesis":
            result = data["data"]["result"]
            assert "cohort_a_data" not in result
            assert "cohort_b_data" not in result


def test_storage_version_check_accepts_version_1(tmp_path: Path):
    """Loading version 1 succeeds."""
    assert 1 in SUPPORTED_VERSIONS

    hypothesis = _make_hypothesis()
    assert hypothesis.spec.version == 1

    store = HypothesisStore(tmp_path)
    store.save(hypothesis)

    loaded = store.load()
    assert len(loaded) == 1


def test_storage_version_check_rejects_unsupported_version(tmp_path: Path):
    """Loading unsupported version raises ValueError."""
    store = HypothesisStore(tmp_path)

    hypothesis = _make_hypothesis()
    store.save(hypothesis)

    session_file = tmp_path / "hypotheses.jsonl"
    lines = session_file.read_text().splitlines()

    modified_lines = []
    for line in lines:
        data = json.loads(line)
        if data.get("record_type") == "hypothesis":
            data["data"]["spec"]["version"] = 99
        modified_lines.append(json.dumps(data, default=str))

    session_file.write_text("\n".join(modified_lines) + "\n")

    with pytest.raises(ValueError) as exc_info:
        store.load()

    assert "Unsupported CyclHyp version: 99" in str(exc_info.value)


def test_storage_empty_session_returns_empty_list(tmp_path: Path):
    """Loading non-existent session returns empty list."""
    store = HypothesisStore(tmp_path)
    loaded = store.load()
    assert loaded == []


def test_config_serialization(tmp_path: Path):
    """to_config_dict() excludes api_key, serializes Path/datetime as strings."""
    from msk_cycl.workflow.session import SessionConfig

    config = SessionConfig(
        parquet_dir=tmp_path / "data",
        storage_dir=tmp_path / "output",
        api_key="secret-key",
    )
    d = config.to_config_dict()

    assert "api_key" not in d
    assert "session_id" in d
    assert "parquet_dir" in d
    assert "created_at" in d
    assert isinstance(d["parquet_dir"], str)
    assert isinstance(d["created_at"], str)
