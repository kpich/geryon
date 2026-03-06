"""Tests for LLM ranker."""

import json
from unittest.mock import Mock

import pandas as pd  # type: ignore

from geryon.labeling.labels import HypothesisRating
from geryon.labeling.models import LabeledHypothesis
from geryon.lang.methods import ComparisonMethod
from geryon.lang.outcomes import OverallSurvival
from geryon.lang.results import ComparisonResult
from geryon.lang.spec import CohortFilter, CompareCohorts, GeryonHyp, SelectCohort
from geryon.llm.generator import HypothesisProposal
from geryon.llm.narrator import HypothesisNarrative
from geryon.llm.ranker import HypothesisRanker, RankerResult


def _make_hyp(
    hypothesis_id: str = "h1",
    novelty: int = 2,
    uncontrolled: int = 2,
    trustworthiness: int = 2,
    notes: str | None = None,
    cohort_a: str = "Group A",
    cohort_b: str = "Group B",
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
    hyp = LabeledHypothesis(
        hypothesis_id=hypothesis_id,
        session_id="s1",
        proposal=HypothesisProposal(
            cohort_a_description=cohort_a,
            cohort_b_description=cohort_b,
            outcome_description="Overall survival",
            rationale="Test rationale",
            geryon_spec=spec,
        ),
        spec=spec,
        result=ComparisonResult(
            cohort_a_size=50,
            cohort_b_size=40,
            hazard_ratio=0.72,
            p_value=0.03,
            confidence_interval_lower=0.55,
            confidence_interval_upper=0.95,
            cohort_a_data=pd.DataFrame({"OS_MONTHS": [10.0], "OS_STATUS": [1]}),
            cohort_b_data=pd.DataFrame({"OS_MONTHS": [8.0], "OS_STATUS": [1]}),
        ),
        execution_time_seconds=1.0,
        narrative=HypothesisNarrative(
            summary="Promising survival difference",
            findings="Significant reduction in hazard",
            limitations=["Small sample"],
            clinical_relevance="Relevant",
        ),
        llm_model="test-model",
        notes=notes,
    )
    hyp.rating = HypothesisRating(
        novelty=novelty,
        uncontrolled=uncontrolled,
        trustworthiness=trustworthiness,
    )
    hyp.labeled_by = "llm_critic"
    return hyp


def test_rank_returns_correct_ids():
    hyps = [_make_hyp("aabbccdd-0000-0000-0000-000000000001")]
    response_json = json.dumps(
        {
            "top_general": ["aabbccdd-0000-0000-0000-000000000001"],
            "top_refinement": [],
            "synthesis": "One promising finding in the dataset.",
        }
    )
    mock_provider = Mock()
    mock_provider.generate.return_value = Mock(
        content=response_json, usage={"total_tokens": 200}
    )

    ranker = HypothesisRanker(mock_provider)
    result = ranker.rank(hyps)

    assert result.top_general == ["aabbccdd-0000-0000-0000-000000000001"]
    assert result.top_refinement == []
    assert "promising" in result.synthesis.lower()


def test_rank_prefix_resolution():
    """Short 8-char IDs from the LLM are resolved to full IDs."""
    full_id = "aabbccdd-0000-0000-0000-000000000001"
    hyps = [_make_hyp(full_id)]
    response_json = json.dumps(
        {
            "top_general": ["aabbccdd"],  # short prefix
            "top_refinement": [],
            "synthesis": "Test synthesis.",
        }
    )
    mock_provider = Mock()
    mock_provider.generate.return_value = Mock(content=response_json, usage=None)

    ranker = HypothesisRanker(mock_provider)
    result = ranker.rank(hyps)

    assert result.top_general == [full_id]


def test_rank_fallback_on_parse_failure():
    hyps = [_make_hyp(f"id{i:04d}") for i in range(15)]
    mock_provider = Mock()
    mock_provider.generate.return_value = Mock(content="not valid json {{{", usage=None)

    ranker = HypothesisRanker(mock_provider)
    result = ranker.rank(hyps)

    assert len(result.top_general) == 10
    assert result.top_refinement == []
    assert result.synthesis == "parse failure"


def test_rank_empty_list():
    ranker = HypothesisRanker(Mock())
    result = ranker.rank([])
    assert result.top_general == []
    assert result.top_refinement == []
    assert "no hypotheses" in result.synthesis.lower()


def test_format_for_prompt_sections():
    full_id_1 = "aaaaaaaa-0000-0000-0000-000000000001"
    full_id_2 = "bbbbbbbb-0000-0000-0000-000000000002"
    hyp1 = _make_hyp(
        full_id_1,
        novelty=3,
        uncontrolled=1,
        trustworthiness=3,
        cohort_a="KRAS mutant",
        cohort_b="KRAS wild-type",
    )
    hyp2 = _make_hyp(
        full_id_2,
        novelty=3,
        uncontrolled=3,
        trustworthiness=2,
        notes="Suggested fix: restrict to Stage 4",
        cohort_a="MSI-H patients",
        cohort_b="MSS patients",
    )

    result = RankerResult(
        top_general=[full_id_1],
        top_refinement=[full_id_2],
        synthesis="Strong KRAS signal detected. MSI-H result is confounded.",
    )
    hyps_by_id = {full_id_1: hyp1, full_id_2: hyp2}

    ranker = HypothesisRanker(Mock())
    text = ranker.format_for_prompt(result, hyps_by_id)

    assert "**SYNTHESIS OF CURRENT FINDINGS:**" in text
    assert "Strong KRAS signal" in text
    assert "**TOP CANDIDATES TO EXPLORE FURTHER:**" in text
    assert "aaaaaaaa" in text
    assert "KRAS mutant vs KRAS wild-type" in text
    assert "novelty=3" in text
    assert "**TOP CANDIDATES FOR REFINEMENT (confounded, worth fixing):**" in text
    assert "bbbbbbbb" in text
    assert "Suggested fix: restrict to Stage 4" in text


def test_format_for_prompt_missing_id():
    result = RankerResult(
        top_general=["deadbeef-0000-0000-0000-000000000099"],
        top_refinement=[],
        synthesis="Test.",
    )
    ranker = HypothesisRanker(Mock())
    text = ranker.format_for_prompt(result, {})
    assert "not found" in text


def test_build_user_prompt_includes_ratings():
    hyp = _make_hyp("h1", novelty=3, uncontrolled=2, trustworthiness=1)
    ranker = HypothesisRanker(Mock())
    prompt = ranker._build_user_prompt([hyp])
    assert "novelty=3" in prompt
    assert "uncontrolled=2" in prompt
    assert "trust=1" in prompt
    assert "h1" in prompt
    assert "HR=0.720" in prompt
