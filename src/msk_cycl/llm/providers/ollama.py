"""Ollama local model provider."""

import requests

from msk_cycl.llm.providers.base import LLMResponse, Message


class OllamaProvider:
    """Ollama local model provider.

    Connects to local Ollama server for running models like mixtral, llama3, etc.
    """

    def __init__(
        self,
        model: str = "mixtral:8x7b",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize Ollama provider.

        Parameters
        ----------
        model : str
            Ollama model name (e.g., "mixtral:8x7b", "llama3:8b")
        base_url : str
            Ollama server URL
        """
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate completion from messages via Ollama API.

        Parameters
        ----------
        messages : list[Message]
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
        requests.exceptions.ConnectionError
            If Ollama server is not reachable
        requests.exceptions.HTTPError
            If API returns error status
        """
        # Convert to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Call Ollama /api/chat endpoint
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=300,  # 5 minute timeout for long generations
        )

        response.raise_for_status()

        data = response.json()

        # Extract response content
        content = data["message"]["content"]

        # Extract token usage if available
        usage = None
        if "prompt_eval_count" in data and "eval_count" in data:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0)
                + data.get("eval_count", 0),
            }

        return LLMResponse(
            content=content,
            model=self.model_id(),
            usage=usage,
        )

    def model_id(self) -> str:
        """Return model identifier."""
        return f"ollama/{self.model}"
