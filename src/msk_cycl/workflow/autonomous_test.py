"""Tests for format_previous_hypotheses."""

import pandas as pd  # type: ignore

from msk_cycl.labeling.labels import HypothesisRating
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CohortFilter, CompareCohorts, CyclHyp, SelectCohort
from msk_cycl.llm.generator import HypothesisProposal
from msk_cycl.llm.narrator import HypothesisNarrative
from msk_cycl.workflow.context import format_previous_hypotheses


def _make_hyp(
    hypothesis_id: str = "h1",
    session_id: str = "s1",
    a_desc: str = "Group A",
    b_desc: str = "Group B",
    rating: HypothesisRating | None = None,
    notes: str | None = None,
) -> LabeledHypothesis:
    spec = CyclHyp(
        query=CompareCohorts(
            cohort_a=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="TX",
                        operator="==",
                        value="A",
                    )
                ]
            ),
            cohort_b=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="TX",
                        operator="==",
                        value="B",
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
            cohort_a_description=a_desc,
            cohort_b_description=b_desc,
            outcome_description="Overall survival",
            rationale="Test",
            cycl_spec=spec,
        ),
        spec=spec,
        result=ComparisonResult(
            cohort_a_size=50,
            cohort_b_size=40,
            cohort_a_data=pd.DataFrame({"OS_MONTHS": [10.0], "OS_STATUS": [1]}),
            cohort_b_data=pd.DataFrame({"OS_MONTHS": [8.0], "OS_STATUS": [1]}),
        ),
        execution_time_seconds=1.0,
        narrative=HypothesisNarrative(
            summary="Test",
            findings="Test",
            limitations=["Lim"],
            clinical_relevance="Rel",
        ),
        llm_model="test-model",
        rating=rating or HypothesisRating(),
        notes=notes,
    )


def test_empty_inputs():
    result = format_previous_hypotheses([], [])
    assert result == "**No previous hypotheses yet.**"


def test_rated_only():
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=3, trustworthiness=3),
        ),
        _make_hyp(
            "h2",
            a_desc="Lung",
            b_desc="Non-lung",
            rating=HypothesisRating(uncontrolled=3, is_duplicate=True),
            notes="used wrong column",
        ),
    ]
    result = format_previous_hypotheses(labeled, [])

    assert "PREVIOUSLY RATED" in result
    assert "KRAS mut vs WT [novelty=3, trust=3]" in result
    assert "Lung vs Non-lung [uncontrolled=3, dup] — used wrong column" in result
    assert "PREVIOUSLY TESTED THIS SESSION" not in result


def test_session_only():
    session = [_make_hyp("s1", a_desc="TP53 mut", b_desc="WT")]
    result = format_previous_hypotheses([], session)

    assert "PREVIOUSLY TESTED THIS SESSION" in result
    assert "TP53 mut vs WT" in result
    assert "PREVIOUSLY RATED" not in result


def test_both_sections():
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=1),
        ),
    ]
    session = [_make_hyp("s1", a_desc="TP53 mut", b_desc="WT")]
    result = format_previous_hypotheses(labeled, session)

    assert "PREVIOUSLY RATED" in result
    assert "PREVIOUSLY TESTED THIS SESSION" in result
    lines = result.split("\n")
    rated_idx = next(i for i, line in enumerate(lines) if "RATED" in line)
    session_idx = next(i for i, line in enumerate(lines) if "THIS SESSION" in line)
    assert rated_idx < session_idx


def test_deduplication():
    """Session hypothesis already in labeled set is excluded from session section."""
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=2),
        ),
    ]
    session = [
        _make_hyp("h1", a_desc="KRAS mut", b_desc="WT"),
        _make_hyp("s2", a_desc="TP53 mut", b_desc="WT"),
    ]
    result = format_previous_hypotheses(labeled, session)

    assert result.count("KRAS mut vs WT") == 1
    assert "TP53 mut vs WT" in result


def test_pending_labeled_excluded_from_rated_section():
    """Labeled hypotheses with pending rating are not shown in the rated section."""
    labeled = [
        _make_hyp("h1", a_desc="KRAS mut", b_desc="WT"),
    ]
    result = format_previous_hypotheses(labeled, [])

    assert result == "**No previous hypotheses yet.**"


def test_truncation_rated():
    labeled = [
        _make_hyp(
            f"h{i}",
            a_desc=f"Gene{i} mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=1),
        )
        for i in range(60)
    ]
    result = format_previous_hypotheses(labeled, [])

    assert "... and 10 more rated hypotheses" in result
    assert "Gene49" in result
    assert "Gene50" not in result


def test_truncation_session():
    session = [
        _make_hyp(f"s{i}", a_desc=f"Gene{i} mut", b_desc="WT") for i in range(25)
    ]
    result = format_previous_hypotheses([], session)

    assert "... and 5 more" in result
    assert "Gene19" in result
    assert "Gene20" not in result


def test_numbering_is_continuous():
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=2, trustworthiness=3),
        ),
        _make_hyp(
            "h2",
            a_desc="BRAF mut",
            b_desc="WT",
            rating=HypothesisRating(uncontrolled=2),
        ),
    ]
    session = [_make_hyp("s1", a_desc="TP53 mut", b_desc="WT")]
    result = format_previous_hypotheses(labeled, session)

    assert "1. KRAS mut vs WT [novelty=2, trust=3]" in result
    assert "2. BRAF mut vs WT [uncontrolled=2]" in result
    assert "3. TP53 mut vs WT" in result
