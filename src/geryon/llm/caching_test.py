"""Tests for prompt-caching helpers."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from geryon.llm.caching import (
    cached_text_content,
    supports_cache_control,
    tail_cache_pre_model_hook,
)


def test_supports_cache_control():
    assert supports_cache_control("aws_bedrock")
    assert supports_cache_control("anthropic")
    assert not supports_cache_control("openai")
    assert not supports_cache_control("unknown")


def test_cached_text_content_marks_breakpoint():
    blocks = cached_text_content("hello")
    assert blocks == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]


def test_tail_hook_marks_last_tool_result():
    state = {
        "messages": [
            SystemMessage(content=cached_text_content("sys")),
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"id": "t1", "name": "q", "args": {}}]),
            ToolMessage(content="big result", tool_call_id="t1"),
        ]
    }
    out = tail_cache_pre_model_hook(state)
    msgs = out["llm_input_messages"]
    block = msgs[-1].content[0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1"
    assert block["content"] == "big result"
    assert block["cache_control"] == {"type": "ephemeral"}
    # earlier messages untouched
    assert msgs[1].content == "hi"


def test_tail_hook_leaves_non_tool_tail_untouched():
    state = {
        "messages": [
            SystemMessage(content=cached_text_content("sys")),
            HumanMessage(content=cached_text_content("hi")),
        ]
    }
    out = tail_cache_pre_model_hook(state)
    assert out["llm_input_messages"][-1].content == cached_text_content("hi")


def test_tail_hook_empty_messages():
    assert tail_cache_pre_model_hook({"messages": []}) == {"llm_input_messages": []}
