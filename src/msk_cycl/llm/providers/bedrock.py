"""AWS Bedrock provider."""

from typing import Any

import boto3

from msk_cycl.llm.providers.base import ChatMessage, LLMResponse


class BedrockProvider:
    """AWS Bedrock provider using the Converse API."""

    def __init__(
        self,
        model: str,
        region: str | None = None,
        profile: str | None = None,
    ):
        self.model = model
        self.region = region or "us-east-1"

        session = boto3.Session(
            profile_name=profile,
            region_name=region,
        )
        self.client = session.client("bedrock-runtime", region_name=self.region)

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        cache_system: bool = False,
    ) -> LLMResponse:
        system_prompts: list[dict[str, Any]] = []
        converse_messages = []

        for msg in messages:
            if msg.role == "system":
                system_prompts.append({"text": msg.content})
            else:
                converse_messages.append(
                    {
                        "role": msg.role,
                        "content": [{"text": msg.content}],
                    }
                )

        kwargs = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system_prompts:
            if cache_system:
                system_prompts.append({"cachePoint": {"type": "default"}})
            kwargs["system"] = system_prompts

        response = self.client.converse(**kwargs)

        content = response["output"]["message"]["content"][0]["text"]

        usage = None
        if "usage" in response:
            usage = {
                "prompt_tokens": response["usage"]["inputTokens"],
                "completion_tokens": response["usage"]["outputTokens"],
                "total_tokens": response["usage"]["totalTokens"],
                "cache_read_tokens": response["usage"].get(
                    "cacheReadInputTokenCount", 0
                ),
                "cache_write_tokens": response["usage"].get(
                    "cacheWriteInputTokenCount", 0
                ),
            }

        return LLMResponse(
            content=content,
            model=self.model_id(),
            usage=usage,
        )

    def model_id(self) -> str:
        return f"bedrock/{self.model}"
