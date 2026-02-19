"""Tests for format_previous_hypotheses."""

import pandas as pd  # type: ignore

from geryon.labeling.labels import HypothesisRating
from geryon.labeling.models import LabeledHypothesis
from geryon.lang.methods import ComparisonMethod
from geryon.lang.outcomes import OverallSurvival
from geryon.lang.results import ComparisonResult
from geryon.lang.spec import CohortFilter, CompareCohorts, GeryonHyp, SelectCohort
from geryon.llm.generator import HypothesisProposal
from geryon.llm.narrator import HypothesisNarrative
from geryon.workflow.context import (
    format_previous_hypotheses,
    load_prior_hypotheses,
    short_id,
)


def _make_hyp(
    hypothesis_id: str = "h1",
    session_id: str = "s1",
    a_desc: str = "Group A",
    b_desc: str = "Group B",
    rating: HypothesisRating | None = None,
    notes: str | None = None,
    summary: str = "Test",
    refines_hypothesis: str | None = None,
) -> LabeledHypothesis:
    spec = GeryonHyp(
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
            geryon_spec=spec,
            refines_hypothesis=refines_hypothesis,
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
            summary=summary,
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
    assert "[h1] KRAS mut vs WT [novelty=3, trust=3] [auto]" in result
    assert (
        "[h2] Lung vs Non-lung [uncontrolled=3, dup] [auto] — used wrong column | Test"
        in result
    )
    assert "PREVIOUSLY TESTED" not in result


def test_session_only():
    session = [_make_hyp("s1", a_desc="TP53 mut", b_desc="WT")]
    result = format_previous_hypotheses([], session)

    assert "PREVIOUSLY TESTED" in result
    assert "[s1] TP53 mut vs WT" in result
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
    assert "PREVIOUSLY TESTED" in result
    lines = result.split("\n")
    rated_idx = next(i for i, line in enumerate(lines) if "RATED" in line)
    session_idx = next(i for i, line in enumerate(lines) if "PREVIOUSLY TESTED" in line)
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
            f"h{i:07d}",
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
        _make_hyp(f"s{i:07d}", a_desc=f"Gene{i} mut", b_desc="WT") for i in range(25)
    ]
    result = format_previous_hypotheses([], session)

    assert "... and 5 more" in result
    assert "Gene19" in result
    assert "Gene20" not in result


def test_session_hypothesis_with_critic_rating_shows_tag():
    """Session hypothesis with critic rating shows tag and [critic] source."""
    session = [
        _make_hyp(
            "s1",
            a_desc="EGFR mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=3, trustworthiness=2),
            notes="interesting",
        ),
    ]
    session[0].labeled_by = "llm_critic"
    result = format_previous_hypotheses([], session)

    assert "EGFR mut vs WT" in result
    assert "[novelty=3, trust=2]" in result
    assert "[critic]" in result
    assert "interesting" in result


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

    assert "1. [h1] KRAS mut vs WT [novelty=2, trust=3] [auto]" in result
    assert "2. [h2] BRAF mut vs WT [uncontrolled=2] [auto]" in result
    assert "3. [s1] TP53 mut vs WT" in result


def test_short_id_shown_in_context():
    """8-char short ID appears in brackets in output."""
    hyp_id = "a3f1c9e2-1234-5678-9abc-def012345678"
    labeled = [
        _make_hyp(
            hyp_id,
            a_desc="KRAS mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=2),
        ),
    ]
    result = format_previous_hypotheses(labeled, [])

    assert "[a3f1c9e2]" in result
    assert short_id(hyp_id) == "a3f1c9e2"


def test_narrative_summary_shown_in_context():
    """Narrative summary appears as | suffix."""
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=2),
            summary="HR=0.72 suggesting protective effect",
        ),
    ]
    result = format_previous_hypotheses(labeled, [])

    assert "| HR=0.72 suggesting protective effect" in result


def test_refines_tag_shown_in_context():
    """(refines xxxxxxxx) tag shown when refines_hypothesis is set."""
    parent_id = "a3f1c9e2-1234-5678-9abc-def012345678"
    labeled = [
        _make_hyp(
            "b7d4e1f0-aaaa-bbbb-cccc-ddddeeeefffff",
            a_desc="KRAS mut age>=60",
            b_desc="WT age>=60",
            rating=HypothesisRating(novelty=3, trustworthiness=3),
            refines_hypothesis=parent_id,
        ),
    ]
    result = format_previous_hypotheses(labeled, [])

    assert "(refines a3f1c9e2)" in result


def test_refines_hypothesis_defaults_to_none():
    """HypothesisProposal without refines_hypothesis defaults to None."""
    spec = GeryonHyp(
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
    proposal = HypothesisProposal(
        cohort_a_description="A",
        cohort_b_description="B",
        outcome_description="OS",
        rationale="test",
        geryon_spec=spec,
    )
    assert proposal.refines_hypothesis is None


def test_summary_shown_in_session_section():
    """Narrative summary shown for unrated session hypotheses too."""
    session = [
        _make_hyp("s1", a_desc="KRAS mut", b_desc="WT", summary="No difference"),
    ]
    result = format_previous_hypotheses([], session)

    assert "| No difference" in result


def test_refines_tag_in_session_section():
    """(refines ...) tag shown in session section."""
    parent_id = "deadbeef-1234-5678-9abc-def012345678"
    session = [
        _make_hyp(
            "c0ffee00-aaaa-bbbb-cccc-ddddeeeefffff",
            a_desc="KRAS mut, NSCLC",
            b_desc="WT, NSCLC",
            refines_hypothesis=parent_id,
        ),
    ]
    result = format_previous_hypotheses([], session)

    assert "(refines deadbeef)" in result


def test_prior_session_hypotheses_loaded(tmp_path):
    """Prior session hypotheses are loaded and included in context."""
    from geryon.labeling.storage import HypothesisStore

    prior_dir = tmp_path / "2024-01-01" / "prior-session-id"
    prior_dir.mkdir(parents=True)
    prior_hyp = _make_hyp(
        "prior-h1", session_id="prior-session-id", a_desc="BRAF mut", b_desc="WT"
    )
    store = HypothesisStore(prior_dir)
    store.save(prior_hyp)

    result = load_prior_hypotheses(tmp_path, "current-session-id")

    assert len(result) == 1
    assert result[0].hypothesis_id == "prior-h1"
    assert result[0].proposal.cohort_a_description == "BRAF mut"


def test_prior_hypotheses_excludes_current_session(tmp_path):
    """Hypotheses from the current session are not returned by load_prior_hypotheses."""
    from geryon.labeling.storage import HypothesisStore

    session_dir = tmp_path / "2024-01-01" / "my-session"
    session_dir.mkdir(parents=True)
    hyp = _make_hyp("h1", session_id="my-session", a_desc="TP53 mut", b_desc="WT")
    store = HypothesisStore(session_dir)
    store.save(hyp)

    result = load_prior_hypotheses(tmp_path, "my-session")

    assert len(result) == 0


def test_human_rating_preferred_over_critic():
    """effective_rating returns human values when both human and critic are set."""
    hyp = _make_hyp(
        "h1",
        a_desc="KRAS mut",
        b_desc="WT",
        rating=HypothesisRating(novelty=1, trustworthiness=1),
    )
    hyp.labeled_by = "llm_critic"
    hyp.human_rating = HypothesisRating(novelty=3, trustworthiness=3)

    assert hyp.effective_rating.novelty == 3
    assert hyp.effective_rating.trustworthiness == 3

    labeled = [hyp]
    result = format_previous_hypotheses(labeled, [])
    assert "[novelty=3, trust=3]" in result
    assert "[human]" in result


def test_effective_rating_falls_back_to_critic():
    """effective_rating falls back to rating when human_rating is None."""
    hyp = _make_hyp(
        "h1",
        a_desc="KRAS mut",
        b_desc="WT",
        rating=HypothesisRating(novelty=2, trustworthiness=2),
    )
    hyp.labeled_by = "llm_critic"

    assert hyp.effective_rating.novelty == 2
    assert hyp.human_rating is None

    labeled = [hyp]
    result = format_previous_hypotheses(labeled, [])
    assert "[novelty=2, trust=2]" in result
    assert "[critic]" in result
