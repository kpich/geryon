"""Tests for prompt-caching helpers."""

from geryon.llm.caching import cached_text_content, supports_cache_control


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
