from pathlib import Path
from unittest.mock import patch

from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel
import pytest

from geryon.codeflow.chat import (
    RefusalFallbackChat,
    _bedrock,
    build_chat_model,
    parse_reply,
    reply_text,
)
from geryon.workflow.session import SessionConfig


def _config(**kw) -> SessionConfig:
    return SessionConfig(parquet_dir=Path("p"), storage_dir=Path("s"), **kw)


def test_bedrock_chat_model_sends_effort():
    with patch("geryon.codeflow.chat.ChatBedrock") as chat:
        _bedrock(_config(effort="xhigh"), "m")
    model_kwargs = chat.call_args.kwargs["model_kwargs"]
    assert model_kwargs["output_config"] == {"effort": "xhigh"}


def test_effort_defaults_to_medium():
    assert _config().effort == "medium"


def _reply(model: str, stop_reason: str = "end_turn", content="ok") -> AIMessage:
    return AIMessage(
        content=content,
        additional_kwargs={"stop_reason": stop_reason},
        response_metadata={"model_name": model},
    )


class _Fake(Runnable):
    """Records what each model was sent and answers with a canned reply."""

    def __init__(self, reply: AIMessage):
        self.reply = reply
        self.sent: list[list[BaseMessage]] = []

    def invoke(self, input, config=None, **kwargs):
        self.sent.append(input)
        return self.reply


def _chat(primary: _Fake, fallback: _Fake) -> RefusalFallbackChat:
    return RefusalFallbackChat(
        primary=primary,
        fallback=fallback,
        primary_id="new",
        fallback_id="old",
    )


def test_answer_from_primary_never_reaches_fallback():
    primary, fallback = _Fake(_reply("new")), _Fake(_reply("old"))
    reply = _chat(primary, fallback).invoke([HumanMessage("q")])
    assert reply.response_metadata["model_name"] == "new"
    assert fallback.sent == []


def test_refused_call_is_resent_to_fallback():
    primary = _Fake(_reply("new", "refusal", content="partial ans"))
    fallback = _Fake(_reply("old", content="full answer"))
    reply = _chat(primary, fallback).invoke([HumanMessage("q")])
    assert reply.content == "full answer"
    assert reply.response_metadata["model_name"] == "old"
    assert [m.content for m in fallback.sent[0]] == ["q"]


def test_refusal_from_both_models_raises():
    primary, fallback = _Fake(_reply("new", "refusal")), _Fake(_reply("old", "refusal"))
    with pytest.raises(RuntimeError, match="both new and old refused"):
        _chat(primary, fallback).invoke([HumanMessage("q")])


def test_each_model_sees_only_its_own_thinking_blocks():
    thinking = {"type": "thinking", "thinking": "", "signature": "s"}
    text = {"type": "text", "text": "t"}
    history = [
        HumanMessage("q"),
        AIMessage(content=[thinking, text], response_metadata={"model_name": "new"}),
        AIMessage(content=[thinking, text], response_metadata={"model_name": "old"}),
        HumanMessage("more"),
    ]
    primary = _Fake(_reply("new", "refusal"))
    fallback = _Fake(_reply("old"))
    _chat(primary, fallback).invoke(history)
    assert [m.content for m in primary.sent[0][1:3]] == [[thinking, text], [text]]
    assert [m.content for m in fallback.sent[0][1:3]] == [[text], [thinking, text]]


def test_bind_tools_binds_both_models():
    with patch("geryon.codeflow.chat.ChatBedrock.bind_tools") as bind:
        build_chat_model(_config()).bind_tools([])
    assert bind.call_count == 2


def test_fallback_model_is_built_from_config():
    chat = build_chat_model(_config(fallback_model="older"))
    assert (chat.primary_id, chat.fallback_id) == (
        "us.anthropic.claude-opus-5-5",
        "older",
    )
    assert isinstance(chat.fallback, ChatBedrock) and chat.fallback.model_id == "older"


def _text_reply(content, stop_reason: str = "end_turn") -> AIMessage:
    return AIMessage(content=content, additional_kwargs={"stop_reason": stop_reason})


def test_skips_thinking_block_before_text():
    content = [
        {"type": "thinking", "thinking": "", "signature": "s"},
        {"type": "text", "text": "out"},
    ]
    assert reply_text(_text_reply(content), "narrator") == "out"


def test_plain_string_content():
    assert reply_text(_text_reply("out"), "narrator") == "out"


def test_truncated_response_raises():
    with pytest.raises(RuntimeError, match="max_tokens"):
        reply_text(_text_reply("parti", "max_tokens"), "narrator")


def test_refusal_stop_raises_naming_the_reason():
    with pytest.raises(RuntimeError, match="refusal"):
        reply_text(
            _text_reply('{"summary": "s", "findings": "cour', "refusal"), "narrator"
        )


def test_no_text_raises():
    content = [{"type": "thinking", "thinking": "", "signature": "s"}]
    with pytest.raises(RuntimeError, match="thinking"):
        reply_text(_text_reply(content), "narrator")


class _Reply(BaseModel):
    value: int


class _Reply2(BaseModel):
    value: int
    x: str


def test_parse_reply_strips_an_outer_fence():
    assert parse_reply('```json\n{"value": 3}\n```', _Reply, "r").value == 3


def test_parse_reply_takes_a_fenced_block_after_reasoning():
    content = 'I expect about 0.7.\n\n- prior: ...\n\n```json\n{"value": 7}\n```'
    assert parse_reply(content, _Reply, "r").value == 7


def test_parse_reply_takes_unfenced_json_after_reasoning():
    content = (
        "TP53 is adverse on CDK4/6i ({Razavi 2018}); HR roughly 1.5\u20132.0.\n\n"
        '{"value": 7}'
    )
    assert parse_reply(content, _Reply, "r").value == 7


def test_parse_reply_keeps_braces_inside_string_values():
    reply = parse_reply('Note.\n{"value": 7, "x": "a {b} }"}', _Reply2, "r")
    assert reply.x == "a {b} }"


def test_parse_reply_rejects_prose_without_a_closing_json_block():
    with pytest.raises(ValueError, match="unparseable"):
        parse_reply('About 0.7.\n```json\n{"value": 7}\n```\nThen more.', _Reply, "r")


def test_parse_reply_names_the_role_when_unparseable():
    with pytest.raises(ValueError, match="guesser returned unparseable"):
        parse_reply('{"value": "lots"}', _Reply, "guesser")


def test_parse_reply_rejects_a_non_object():
    with pytest.raises(ValueError, match="unparseable"):
        parse_reply("[1, 2]", _Reply, "r")
