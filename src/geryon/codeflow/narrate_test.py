"""Tests for the narrator's request and response handling (no LLM)."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from geryon.codeflow.narrate import CodeNarrator


def _narrator() -> CodeNarrator:
    return CodeNarrator(MagicMock(), "SYSTEM")


def test_parse_fenced_json():
    content = '```json\n{"summary": "s", "findings": "f"}\n```'
    assert _narrator()._parse(content).summary == "s"


def test_parse_keeps_fenced_block_inside_a_string():
    content = '```json\n{"summary": "s", "findings": "a\\n```\\nb"}\n```'
    assert _narrator()._parse(content).findings == "a\n```\nb"


def test_unparseable_output_raises_instead_of_storing_a_placeholder():
    with pytest.raises(ValueError, match="unparseable"):
        _narrator()._parse("Sure! Here's what I found...")


def test_narrate_sends_system_and_user_and_records_usage():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"summary": "s", "findings": "f"}',
        additional_kwargs={"stop_reason": "end_turn"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={"model_name": "fallback-model"},
    )
    narrator = CodeNarrator(llm, "SYSTEM")
    narrative = narrator.narrate(
        description="d", rationale="r", code="c", result=None, stdout=""
    )
    assert narrative.summary == "s"
    assert narrative.model == "fallback-model"
    system, user = llm.invoke.call_args.args[0]
    assert isinstance(system, SystemMessage) and system.content == "SYSTEM"
    assert isinstance(user, HumanMessage) and "# HYPOTHESIS" in user.content
    assert narrator.last_usage is not None
    assert narrator.last_usage.input_tokens == 10
    assert narrator.last_usage.n_llm_calls == 1
