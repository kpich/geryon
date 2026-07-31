"""LLM provider factory."""

from typing import Any, Literal

from geryon.llm.providers.base import LLMProvider
from geryon.llm.providers.bedrock import BedrockProvider
from geryon.llm.providers.openai import OpenAIProvider

ProviderType = Literal["openai", "anthropic", "aws_bedrock"]

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-opus-5"


def create_provider(
    provider_type: ProviderType,
    **kwargs: Any,
) -> LLMProvider:
    """Factory for creating LLM providers.

    Parameters
    ----------
    provider_type : ProviderType
        Type of provider ("openai", "anthropic", "aws_bedrock")
    **kwargs
        Provider-specific configuration

    Returns
    -------
    LLMProvider
        Configured provider instance

    Raises
    ------
    ValueError
        If provider_type is unknown
    NotImplementedError
        If provider is not yet implemented

    Examples
    --------
    >>> provider = create_provider("aws_bedrock", model=DEFAULT_BEDROCK_MODEL)
    """
    if provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        raise NotImplementedError("Anthropic provider not yet implemented")
    elif provider_type == "aws_bedrock":
        return BedrockProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
