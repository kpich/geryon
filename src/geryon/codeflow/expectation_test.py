"""Tests for the blind expectation (no LLM)."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from pydantic import ValidationError
import pytest

from geryon.codeflow.expectation import elicit_expectation
from geryon.codeflow.models import Expectation


def _reply(text: str, model: str = "m") -> AIMessage:
    return AIMessage(
        content=text,
        additional_kwargs={"stop_reason": "end_turn"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": model},
    )


def _elicit(*replies: AIMessage):
    llm = MagicMock()
    llm.invoke.side_effect = list(replies)
    out = elicit_expectation(
        llm,
        title="TP53 PREDICTS shorter TTNT",
        description="TP53-mutant patients did WORSE, HR 1.6",
        effect_size_type="hazard ratio, TP53-mut vs wt",
    )
    return out, llm


def test_expectation_is_asked_from_the_neutral_question_alone():
    (expectation, usage), llm = _elicit(
        _reply(
            '{"question": "In CDK4/6i-treated breast cancer, HR of TTNT, mut vs wt?"}'
        ),
        _reply('{"effect": 1.3, "lower": 1.0, "upper": 1.7}', model="fallback"),
    )
    assert expectation == Expectation(
        question="In CDK4/6i-treated breast cancer, HR of TTNT, mut vs wt?",
        effect=1.3,
        lower=1.0,
        upper=1.7,
        model="fallback",
    )
    assert usage.n_llm_calls == 2 and usage.input_tokens == 20

    first, second = (c.args[0] for c in llm.invoke.call_args_list)
    assert "WORSE" in first[1].content
    blind = second[0].content + second[1].content
    assert "WORSE" not in blind and "PREDICTS" not in blind
    assert "TP53-mut vs wt" in blind


def test_disordered_interval_raises():
    with pytest.raises(ValueError, match="disordered"):
        _elicit(
            _reply('{"question": "q"}'),
            _reply('{"effect": 2.0, "lower": 1.0, "upper": 1.5}'),
        )


def test_unparseable_guess_raises():
    with pytest.raises(ValueError, match="unparseable"):
        _elicit(_reply('{"question": "q"}'), _reply("I'd guess around 1.3"))


def test_expectation_model_rejects_disordered_bounds():
    with pytest.raises(ValidationError):
        Expectation(question="q", effect=0.5, lower=0.6, upper=0.9)
