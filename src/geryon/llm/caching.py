"""Prompt-caching helpers for the LangChain generation path.

Anthropic-family models (direct or via Bedrock) cache the prompt prefix up to a
content block marked with ``cache_control``. We mark the stable prefix — the
system prompt (instructions + schema context) and the initial user message
(previous-hypotheses block) — so it is read from cache on every tool-call
round-trip instead of re-billed as fresh input.

OpenAI caches prefixes automatically and rejects ``cache_control`` blocks, so
caching is only applied for Anthropic-style providers.
"""

# Providers whose models honor Anthropic-style cache_control breakpoints.
CACHE_CONTROL_PROVIDERS = frozenset({"aws_bedrock", "anthropic"})


def supports_cache_control(provider_type: str) -> bool:
    return provider_type in CACHE_CONTROL_PROVIDERS


def cached_text_content(text: str) -> list[dict]:
    """Wrap text as a single content block carrying an ephemeral cache breakpoint."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
