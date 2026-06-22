"""Tests for format_previous_hypotheses and AutonomousWorkflow helpers."""

import json
from unittest.mock import MagicMock

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
    context_summary: str | None = None,
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
            context_summary=context_summary,
        ),
        llm_model="test-model",
        rating=rating or HypothesisRating(),
        notes=notes,
    )


def test_empty_inputs():
    ctx = format_previous_hypotheses([], [])
    assert ctx.text == "**No previous hypotheses yet.**"
    assert ctx.ids == []


def test_single_hypothesis_appears():
    labeled = [_make_hyp("h1", a_desc="KRAS mut", b_desc="WT")]
    ctx = format_previous_hypotheses(labeled, [])
    assert "PREVIOUSLY TESTED HYPOTHESES" in ctx.text
    assert "[h1]" in ctx.text
    assert "KRAS mut vs WT" in ctx.text
    assert ctx.ids == ["h1"]


def test_rated_hypothesis_shows_tag():
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=3, trustworthiness=3),
        ),
    ]
    ctx = format_previous_hypotheses(labeled, [])
    assert "[novelty=3, trust=3]" in ctx.text


def test_notes_shown():
    labeled = [
        _make_hyp(
            "h2",
            a_desc="Lung",
            b_desc="Non-lung",
            rating=HypothesisRating(uncontrolled=3, is_duplicate=True),
            notes="used wrong column",
        ),
    ]
    ctx = format_previous_hypotheses(labeled, [])
    assert "— used wrong column" in ctx.text
    assert "[uncontrolled=3, dup]" in ctx.text


def test_pending_rated_still_included():
    """Pending-rated hypotheses appear in the unified list (no longer excluded)."""
    labeled = [_make_hyp("h1", a_desc="KRAS mut", b_desc="WT")]
    ctx = format_previous_hypotheses(labeled, [])
    assert "[h1]" in ctx.text
    assert ctx.ids == ["h1"]


def test_session_hypothesis_appears():
    session = [_make_hyp("s1", a_desc="TP53 mut", b_desc="WT")]
    ctx = format_previous_hypotheses([], session)
    assert "PREVIOUSLY TESTED HYPOTHESES" in ctx.text
    assert "[s1] TP53 mut vs WT" in ctx.text
    assert ctx.ids == ["s1"]


def test_deduplication():
    """Session hypothesis already in labeled set uses labeled-store version."""
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
    ctx = format_previous_hypotheses(labeled, session)

    assert ctx.text.count("KRAS mut vs WT") == 1
    assert "TP53 mut vs WT" in ctx.text
    assert "[novelty=2]" in ctx.text


def test_truncation():
    labeled = [
        _make_hyp(
            f"h{i:07d}",
            a_desc=f"Gene{i} mut",
            b_desc="WT",
            rating=HypothesisRating(novelty=1),
        )
        for i in range(210)
    ]
    ctx = format_previous_hypotheses(labeled, [])

    assert "... and 10 more" in ctx.text
    assert len(ctx.ids) == 200


def test_short_id_in_brackets():
    hyp_id = "a3f1c9e2-1234-5678-9abc-def012345678"
    labeled = [_make_hyp(hyp_id, a_desc="KRAS mut", b_desc="WT")]
    result = format_previous_hypotheses(labeled, []).text
    assert "[a3f1c9e2]" in result
    assert short_id(hyp_id) == "a3f1c9e2"


def test_context_summary_used_when_present():
    """context_summary replaces cohort desc + narrative summary when available."""
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut bladder on Cisplatin",
            b_desc="WT bladder on Cisplatin",
            summary="HR=0.71 p=0.003",
            context_summary=(
                "KRAS mut vs WT in Bladder on Cisplatin (OS): HR=0.71 p=0.003,"
                " mutants survive longer; confounded by stage."
            ),
        ),
    ]
    result = format_previous_hypotheses(labeled, []).text
    assert "KRAS mut vs WT in Bladder on Cisplatin" in result
    assert "KRAS mut bladder on Cisplatin vs" not in result


def test_narrative_summary_fallback_when_no_context_summary():
    """Falls back to cohort desc + narrative summary when context_summary absent."""
    labeled = [
        _make_hyp(
            "h1",
            a_desc="KRAS mut",
            b_desc="WT",
            summary="HR=0.72 suggesting protective effect",
        ),
    ]
    result = format_previous_hypotheses(labeled, []).text
    assert "KRAS mut vs WT" in result
    assert "| HR=0.72 suggesting protective effect" in result


def test_refines_tag_shown():
    parent_id = "a3f1c9e2-1234-5678-9abc-def012345678"
    labeled = [
        _make_hyp(
            "b7d4e1f0-aaaa-bbbb-cccc-ddddeeeefffff",
            a_desc="KRAS mut age>=60",
            b_desc="WT age>=60",
            rating=HypothesisRating(novelty=3),
            refines_hypothesis=parent_id,
        ),
    ]
    result = format_previous_hypotheses(labeled, []).text
    assert "(refines a3f1c9e2)" in result


def test_refines_hypothesis_defaults_to_none():
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


def test_human_notes_shown():
    hyp = _make_hyp(
        "h1",
        a_desc="KRAS mut",
        b_desc="WT",
        rating=HypothesisRating(novelty=3),
    )
    hyp.human_notes = "interesting follow-up needed"
    result = format_previous_hypotheses([hyp], []).text
    assert "interesting follow-up needed" in result


def test_critic_rated_session_hypothesis_shows_tag():
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
    result = format_previous_hypotheses([], session).text
    assert "EGFR mut vs WT" in result
    assert "[novelty=3, trust=2]" in result
    assert "interesting" in result


def test_newest_first_ordering():
    """Entries are reversed so newest hypotheses appear first."""
    session = [
        _make_hyp("old", a_desc="Old hyp", b_desc="WT"),
        _make_hyp("new", a_desc="New hyp", b_desc="WT"),
    ]
    result = format_previous_hypotheses([], session).text
    old_pos = result.index("[old]")
    new_pos = result.index("[new]")
    assert new_pos < old_pos


def test_human_rating_preferred_over_critic():
    hyp = _make_hyp(
        "h1",
        a_desc="KRAS mut",
        b_desc="WT",
        rating=HypothesisRating(novelty=1, trustworthiness=1),
    )
    hyp.labeled_by = "llm_critic"
    hyp.human_rating = HypothesisRating(novelty=3, trustworthiness=3)

    assert hyp.effective_rating.novelty == 3
    result = format_previous_hypotheses([hyp], []).text
    assert "[novelty=3, trust=3]" in result


def test_effective_rating_falls_back_to_critic():
    hyp = _make_hyp(
        "h1",
        a_desc="KRAS mut",
        b_desc="WT",
        rating=HypothesisRating(novelty=2, trustworthiness=2),
    )
    hyp.labeled_by = "llm_critic"
    assert hyp.effective_rating.novelty == 2
    assert hyp.human_rating is None

    result = format_previous_hypotheses([hyp], []).text
    assert "[novelty=2, trust=2]" in result


def test_prior_session_hypotheses_loaded(tmp_path):
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
    from geryon.labeling.storage import HypothesisStore

    session_dir = tmp_path / "2024-01-01" / "my-session"
    session_dir.mkdir(parents=True)
    hyp = _make_hyp("h1", session_id="my-session", a_desc="TP53 mut", b_desc="WT")
    store = HypothesisStore(session_dir)
    store.save(hyp)

    result = load_prior_hypotheses(tmp_path, "my-session")
    assert len(result) == 0


def test_replay_derived_views_on_init(tmp_path):
    from geryon.workflow.autonomous import AutonomousWorkflow

    views_json = tmp_path / "derived_views.json"
    views_json.write_text(
        json.dumps(
            {
                "derived_foo": "SELECT 1 AS PATIENT_ID",
                "derived_bar": "SELECT 2 AS PATIENT_ID",
            }
        )
    )

    wf = object.__new__(AutonomousWorkflow)
    wf._derived_views_path = views_json
    wf.db = MagicMock()

    wf._replay_derived_views()

    assert wf.db.create_view.call_count == 2
    wf.db.create_view.assert_any_call("derived_foo", "SELECT 1 AS PATIENT_ID")
    wf.db.create_view.assert_any_call("derived_bar", "SELECT 2 AS PATIENT_ID")


def test_save_derived_view_writes_json(tmp_path):
    from geryon.workflow.autonomous import AutonomousWorkflow

    views_path = tmp_path / "derived_views.json"

    wf = object.__new__(AutonomousWorkflow)
    wf._derived_views_path = views_path

    wf._save_derived_view("derived_regimens", "SELECT PATIENT_ID FROM foo")

    assert views_path.exists()
    data = json.loads(views_path.read_text())
    assert data["derived_regimens"] == "SELECT PATIENT_ID FROM foo"

    wf._save_derived_view("derived_other", "SELECT PATIENT_ID FROM bar")
    data = json.loads(views_path.read_text())
    assert "derived_regimens" in data
    assert "derived_other" in data


def test_sum_message_usage_with_cache_details():
    from langchain_core.messages import AIMessage

    from geryon.workflow.autonomous import _sum_message_usage

    cached = AIMessage(
        content="x",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "input_token_details": {"cache_read": 800, "cache_creation": 100},
        },
    )
    # old-style message without cache details and a non-usage message
    legacy = AIMessage(
        content="y",
        usage_metadata={"input_tokens": 200, "output_tokens": 10, "total_tokens": 210},
    )
    no_usage = AIMessage(content="z")

    u = _sum_message_usage([cached, legacy, no_usage])

    assert u.input_tokens == 1200
    assert u.output_tokens == 60
    assert u.total_tokens == 1260
    assert u.cache_read_tokens == 800
    assert u.cache_creation_tokens == 100
    assert u.n_llm_calls == 2
