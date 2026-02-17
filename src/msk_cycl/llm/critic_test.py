"""Tests for LLM critic."""

import json
from unittest.mock import Mock

import pandas as pd  # type: ignore

from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CohortFilter, CompareCohorts, CyclHyp, SelectCohort
from msk_cycl.llm.critic import HypothesisCritic
from msk_cycl.llm.generator import HypothesisProposal
from msk_cycl.llm.narrator import HypothesisNarrative


def _make_hyp(hypothesis_id: str = "h1") -> LabeledHypothesis:
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
        session_id="s1",
        proposal=HypothesisProposal(
            cohort_a_description="Group A",
            cohort_b_description="Group B",
            outcome_description="Overall survival",
            rationale="Test rationale",
            cycl_spec=spec,
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
            summary="Test summary",
            findings="Test findings",
            limitations=["Lim"],
            clinical_relevance="Rel",
        ),
        llm_model="test-model",
    )


def test_parse_valid_response():
    critic = HypothesisCritic(Mock())
    hyps = [_make_hyp("h1")]
    content = json.dumps(
        {
            "ratings": [
                {
                    "hypothesis_id": "h1",
                    "novelty": 3,
                    "uncontrolled": 1,
                    "trustworthiness": 2,
                    "is_duplicate": False,
                    "notes": "interesting finding",
                }
            ]
        }
    )
    ratings = critic._parse_response(content, hyps)
    assert len(ratings) == 1
    assert ratings[0].hypothesis_id == "h1"
    assert ratings[0].novelty == 3
    assert ratings[0].uncontrolled == 1
    assert ratings[0].trustworthiness == 2
    assert ratings[0].is_duplicate is False
    assert ratings[0].notes == "interesting finding"


def test_parse_response_fallback():
    critic = HypothesisCritic(Mock())
    hyps = [_make_hyp("h1"), _make_hyp("h2")]
    ratings = critic._parse_response("not valid json {{{", hyps)
    assert len(ratings) == 2
    assert all(r.novelty == 2 for r in ratings)
    assert all(r.uncontrolled == 2 for r in ratings)
    assert all(r.trustworthiness == 2 for r in ratings)
    assert all("neutral defaults" in r.notes for r in ratings)


def test_parse_response_markdown_wrapped():
    critic = HypothesisCritic(Mock())
    hyps = [_make_hyp("h1")]
    content = (
        '```json\n{"ratings": [{"hypothesis_id": "h1", "novelty": 1, '
        '"uncontrolled": 3, "trustworthiness": 2}]}\n```'
    )
    ratings = critic._parse_response(content, hyps)
    assert len(ratings) == 1
    assert ratings[0].novelty == 1
    assert ratings[0].uncontrolled == 3


def test_rate_applies_ratings():
    mock_provider = Mock()
    mock_provider.generate.return_value = Mock(
        content=json.dumps(
            {
                "ratings": [
                    {
                        "hypothesis_id": "h1",
                        "novelty": 3,
                        "uncontrolled": 1,
                        "trustworthiness": 2,
                        "is_duplicate": False,
                        "notes": "novel",
                    }
                ]
            }
        ),
        usage={"total_tokens": 100},
    )

    critic = HypothesisCritic(mock_provider)
    hyps = [_make_hyp("h1")]
    result = critic.rate(hyps)

    assert result is hyps
    assert hyps[0].rating.novelty == 3
    assert hyps[0].rating.uncontrolled == 1
    assert hyps[0].rating.trustworthiness == 2
    assert hyps[0].labeled_by == "llm_critic"
    assert hyps[0].labeled_at is not None
    assert hyps[0].notes == "novel"


def test_rate_empty_list():
    critic = HypothesisCritic(Mock())
    result = critic.rate([])
    assert result == []


def test_build_user_prompt_includes_all_hypotheses():
    critic = HypothesisCritic(Mock())
    hyps = [_make_hyp("h1"), _make_hyp("h2"), _make_hyp("h3")]
    prompt = critic._build_user_prompt(hyps)
    assert "h1" in prompt
    assert "h2" in prompt
    assert "h3" in prompt
    assert "HR=0.720" in prompt
    assert "p=0.0300" in prompt
