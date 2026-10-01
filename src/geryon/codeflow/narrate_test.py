"""Tests for the narrator's request and response handling (no LLM)."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

from geryon.codeflow.narrate import CodeNarrator, _response_text


def _narrator() -> CodeNarrator:
    return CodeNarrator(MagicMock(), "SYSTEM")


def test_parse_fenced_json():
    content = '```json\n{"summary": "s", "findings": "f"}\n```'
    assert _narrator()._parse(content).summary == "s"


def test_unparseable_output_raises_instead_of_storing_a_placeholder():
    with pytest.raises(ValueError, match="unparseable"):
        _narrator()._parse("Sure! Here's what I found...")


def _reply(content, stop_reason: str = "end_turn") -> AIMessage:
    return AIMessage(content=content, additional_kwargs={"stop_reason": stop_reason})


def test_skips_thinking_block_before_text():
    content = [
        {"type": "thinking", "thinking": "", "signature": "s"},
        {"type": "text", "text": "out"},
    ]
    assert _response_text(_reply(content)) == "out"


def test_plain_string_content():
    assert _response_text(_reply("out")) == "out"


def test_truncated_response_raises():
    with pytest.raises(RuntimeError, match="max_tokens"):
        _response_text(_reply("parti", "max_tokens"))


def test_no_text_raises():
    content = [{"type": "thinking", "thinking": "", "signature": "s"}]
    with pytest.raises(RuntimeError, match="thinking"):
        _response_text(_reply(content))


def test_narrate_sends_system_and_user_and_records_usage():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"summary": "s", "findings": "f"}',
        additional_kwargs={"stop_reason": "end_turn"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    narrator = CodeNarrator(llm, "SYSTEM")
    narrative = narrator.narrate(
        description="d", rationale="r", code="c", result=None, stdout=""
    )
    assert narrative.summary == "s"
    system, user = llm.invoke.call_args.args[0]
    assert isinstance(system, SystemMessage) and system.content == "SYSTEM"
    assert isinstance(user, HumanMessage) and "# HYPOTHESIS" in user.content
    assert narrator.last_usage is not None
    assert narrator.last_usage.input_tokens == 10
    assert narrator.last_usage.n_llm_calls == 1
