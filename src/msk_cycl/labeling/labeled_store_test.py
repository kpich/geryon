"""Tests for labeled hypothesis store."""

import json
from pathlib import Path

import pandas as pd  # type: ignore

from msk_cycl.labeling.labeled_store import LabeledStore
from msk_cycl.labeling.labels import HypothesisRating
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CohortFilter, CompareCohorts, CyclHyp, SelectCohort
from msk_cycl.llm.generator import HypothesisProposal
from msk_cycl.llm.narrator import HypothesisNarrative


def _make_hypothesis(
    hypothesis_id: str = "test-id", session_id: str = "s1"
) -> LabeledHypothesis:
    spec = CyclHyp(
        query=CompareCohorts(
            cohort_a=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient", column="TX", operator="==", value="A"
                    )
                ]
            ),
            cohort_b=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient", column="TX", operator="==", value="B"
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
            cohort_a_description="Group A",
            cohort_b_description="Group B",
            outcome_description="Overall survival",
            rationale="Test",
            cycl_spec=spec,
        ),
        spec=spec,
        result=ComparisonResult(
            cohort_a_size=50,
            cohort_b_size=40,
            hazard_ratio=0.72,
            p_value=0.03,
            cohort_a_data=pd.DataFrame({"OS_MONTHS": [10.0], "OS_STATUS": [1]}),
            cohort_b_data=pd.DataFrame({"OS_MONTHS": [8.0], "OS_STATUS": [1]}),
        ),
        execution_time_seconds=1.0,
        narrative=HypothesisNarrative(
            summary="Test summary",
            findings="Test findings",
            limitations=["Limitation"],
            clinical_relevance="Relevant",
        ),
        llm_model="test-model",
        rating=HypothesisRating(novelty=2, trustworthiness=3),
        notes="Good one",
        labeled_by="tester",
    )


def test_save_load_round_trip(tmp_path: Path):
    store = LabeledStore(tmp_path)
    hyp = _make_hypothesis()
    store.save(hyp)

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].hypothesis_id == "test-id"
    assert loaded[0].rating.novelty == 2
    assert loaded[0].rating.trustworthiness == 3
    assert loaded[0].notes == "Good one"
    assert loaded[0].result.cohort_a_size == 50
    assert loaded[0].result.cohort_b_size == 40


def test_labeled_ids(tmp_path: Path):
    store = LabeledStore(tmp_path)
    store.save(_make_hypothesis("id-1"))
    store.save(_make_hypothesis("id-2"))

    ids = store.labeled_ids()
    assert ids == {"id-1", "id-2"}


def test_bulk_data_stripped_from_json(tmp_path: Path):
    store = LabeledStore(tmp_path)
    store.save(_make_hypothesis())

    content = json.loads((tmp_path / "test-id.json").read_text())
    assert "cohort_a_data" not in content["result"]
    assert "cohort_b_data" not in content["result"]
    assert content["result"]["cohort_a_size"] == 50


def test_creates_directory(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c"
    LabeledStore(nested)
    assert nested.exists()


def test_empty_store_returns_empty(tmp_path: Path):
    store = LabeledStore(tmp_path)
    assert store.load_all() == []
    assert store.labeled_ids() == set()
