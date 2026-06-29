"""Prompt-caching helpers for the LangChain generation path.

Anthropic-family models (direct or via Bedrock) cache the prompt prefix up to a
content block marked with ``cache_control``. We mark the stable prefix — the
system prompt (instructions + schema context) and the initial user message
(previous-hypotheses block) — so it is read from cache on every tool-call
round-trip instead of re-billed as fresh input.

The stable prefix alone leaves the largest cost — the growing tool-result tail
re-sent on every round-trip — uncached. ``tail_cache_pre_model_hook`` adds a
second, sliding breakpoint on the most recent tool result before each LLM call,
so the entire conversation prefix (system + prior turns + tool outputs) is read
from cache rather than re-billed as fresh input.

OpenAI caches prefixes automatically and rejects ``cache_control`` blocks, so
caching is only applied for Anthropic-style providers.
"""

from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

# Providers whose models honor Anthropic-style cache_control breakpoints.
CACHE_CONTROL_PROVIDERS = frozenset({"aws_bedrock", "anthropic"})

_EPHEMERAL = {"type": "ephemeral"}


def supports_cache_control(provider_type: str) -> bool:
    return provider_type in CACHE_CONTROL_PROVIDERS


def cached_text_content(text: str) -> list[str | dict[str, Any]]:
    """Wrap text as a single content block carrying an ephemeral cache breakpoint.

    Return type matches LangChain message ``content`` (``str | list[str | dict]``)
    so it can be passed straight to SystemMessage/HumanMessage without a cast.
    """
    return [{"type": "text", "text": text, "cache_control": _EPHEMERAL}]


def _with_tail_breakpoint(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return messages with a cache breakpoint on the latest tool result.

    Marks the final ``ToolMessage`` by reshaping its content into a
    ``tool_result`` block carrying ``cache_control`` (the only shape the Bedrock
    Anthropic formatter preserves the breakpoint on). Any other trailing message
    type — e.g. the initial human turn, already cached at construction — is left
    untouched. Returns the list unchanged on anything unexpected so a caching
    quirk can never break a generation run.
    """
    if not messages:
        return messages
    last = messages[-1]
    if not isinstance(last, ToolMessage) or not isinstance(last.content, str):
        return messages
    block = {
        "type": "tool_result",
        "tool_use_id": last.tool_call_id,
        "content": last.content,
        "cache_control": _EPHEMERAL,
    }
    return [*messages[:-1], last.model_copy(update={"content": [block]})]


def tail_cache_pre_model_hook(state: dict) -> dict:
    """create_react_agent pre-model hook: slide a cache breakpoint to the tail.

    Returns the marked messages via ``llm_input_messages`` so the breakpoint is
    applied only to what the LLM sees, without mutating graph state.
    """
    messages = state.get("messages", [])
    return {"llm_input_messages": _with_tail_breakpoint(list(messages))}
