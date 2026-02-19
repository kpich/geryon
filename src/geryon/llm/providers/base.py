"""Base types and protocols for LLM providers."""

from typing import Literal, Protocol

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """Response from LLM provider."""

    content: str
    model: str
    usage: dict[str, int] | None = Field(
        default=None, description="Token usage statistics"
    )


class LLMProvider(Protocol):
    """Abstract interface for LLM providers.

    Implementations must support message-based generation with temperature control.
    """

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        cache_system: bool = False,
    ) -> LLMResponse:
        """Generate completion from messages.

        Parameters
        ----------
        messages : list[ChatMessage]
            Conversation history
        temperature : float
            Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens : int
            Maximum tokens to generate

        Returns
        -------
        LLMResponse
            Generated response with content and metadata
        """
        ...

    def model_id(self) -> str:
        """Return model identifier for provenance tracking."""
        ...
