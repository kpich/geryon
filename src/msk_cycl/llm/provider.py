"""LLM provider factory."""

from typing import Any, Literal

from msk_cycl.llm.providers.base import LLMProvider
from msk_cycl.llm.providers.bedrock import BedrockProvider
from msk_cycl.llm.providers.ollama import OllamaProvider
from msk_cycl.llm.providers.openai import OpenAIProvider

ProviderType = Literal["ollama", "openai", "anthropic", "aws_bedrock"]


def create_provider(
    provider_type: ProviderType,
    **kwargs: Any,
) -> LLMProvider:
    """Factory for creating LLM providers.

    Parameters
    ----------
    provider_type : ProviderType
        Type of provider ("ollama", "openai", "anthropic")
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
    >>> provider = create_provider("ollama", model="mixtral:8x7b")
    >>> provider = create_provider("ollama", model="llama3:8b", base_url="http://localhost:11434")
    """
    if provider_type == "ollama":
        return OllamaProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "anthropic":
        raise NotImplementedError("Anthropic provider not yet implemented")
    elif provider_type == "aws_bedrock":
        return BedrockProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
