"""Chat models for the generator and critic, and token-usage accounting for all
three LLM phases."""

from typing import NamedTuple

from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock

from geryon.llm.providers.base import LLMResponse
from geryon.workflow.session import SessionConfig


class MessageUsage(NamedTuple):
    """Token usage summed over one LLM phase (generation, critic, narration)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    n_llm_calls: int


def sum_message_usage(messages: list) -> MessageUsage:
    """Sum token usage across a LangGraph/LangChain message list.

    Reads each message's ``usage_metadata`` (set on AI responses). On Bedrock,
    ``input_tokens`` is the uncached input only and the ``input_token_details`` cache
    counts are separate from it, the same disjoint buckets the cost plot prices.
    """
    inp = out = tot = cache_read = cache_create = calls = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        i = int(usage.get("input_tokens", 0) or 0)
        o = int(usage.get("output_tokens", 0) or 0)
        inp += i
        out += o
        tot += int(usage.get("total_tokens", 0) or 0) or (i + o)
        details = usage.get("input_token_details") or {}
        cache_read += int(details.get("cache_read", 0) or 0)
        cache_create += int(details.get("cache_creation", 0) or 0)
        calls += 1
    return MessageUsage(inp, out, tot, cache_read, cache_create, calls)


def usage_from_response(response: LLMResponse) -> MessageUsage:
    """Map one provider ``LLMResponse.usage`` dict into a ``MessageUsage``.

    ``prompt_tokens`` is the uncached input; ``cache_read_tokens`` and
    ``cache_write_tokens`` are separate from it, as the cost plot expects.
    """
    usage = response.usage or {}
    inp = int(usage.get("prompt_tokens", 0) or 0)
    out = int(usage.get("completion_tokens", 0) or 0)
    tot = int(usage.get("total_tokens", 0) or 0) or (inp + out)
    return MessageUsage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=tot,
        cache_read_tokens=int(usage.get("cache_read_tokens", 0) or 0),
        cache_creation_tokens=int(usage.get("cache_write_tokens", 0) or 0),
        n_llm_calls=1,
    )


def build_chat_model(config: SessionConfig) -> ChatBedrock:
    """Build the LangChain Bedrock chat model the generator and critic run on."""
    boto_config = BotoConfig(
        read_timeout=300, connect_timeout=30, retries={"max_attempts": 2}
    )
    kwargs: dict = {
        "model_id": config.model,
        # Current Claude models 400 on temperature/top_p/top_k, so none are sent.
        "model_kwargs": {
            "max_tokens": 16384,
            "output_config": {"effort": config.effort},
        },
        "config": boto_config,
    }
    if config.aws_region:
        kwargs["region_name"] = config.aws_region
    if config.aws_profile:
        kwargs["credentials_profile_name"] = config.aws_profile
    if "arn:" in config.model or "anthropic" in config.model.lower():
        kwargs["provider"] = "anthropic"
    return ChatBedrock(**kwargs)
