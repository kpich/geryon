"""OpenAI API provider."""

import os

from openai import OpenAI

from msk_cycl.llm.providers.base import ChatMessage, LLMResponse


class OpenAIProvider:
    """OpenAI API provider with support for custom endpoints.

    Supports both official OpenAI API and OpenAI-compatible endpoints
    (e.g., AWS Bedrock proxy, vLLM, Text Generation Inference).
    """

    def __init__(
        self,
        model: str = "gpt-4",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize OpenAI provider.

        Parameters
        ----------
        model : str
            Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
        base_url : str, optional
            Custom API endpoint. If None, uses default OpenAI endpoint.
        api_key : str, optional
            API key for authentication. If None, reads from OPENAI_API_KEY env var.
        """
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate completion from messages via OpenAI API.

        Parameters
        ----------
        messages : list[ChatMessage]
            Conversation history
        temperature : float
            Sampling temperature
        max_tokens : int
            Maximum tokens to generate

        Returns
        -------
        LLMResponse
            Generated response

        Raises
        ------
        openai.OpenAIError
            If API request fails
        """
        openai_messages: list[dict[str, str]] = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=self.model_id(),
            usage=usage,
        )

    def model_id(self) -> str:
        """Return model identifier."""
        return f"openai/{self.model}"
