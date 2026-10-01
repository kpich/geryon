"""AWS Bedrock provider."""

from typing import Any

import boto3

from geryon.llm.providers.base import ChatMessage, LLMResponse


class BedrockProvider:
    """AWS Bedrock provider using the Converse API."""

    def __init__(
        self,
        model: str,
        region: str | None = None,
        profile: str | None = None,
        effort: str | None = None,
    ):
        self.model = model
        self.region = region
        self.effort = effort

        session = boto3.Session(
            profile_name=profile,
            region_name=region,
        )
        self.client = session.client("bedrock-runtime", region_name=self.region)

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 32000,
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

        # temperature is accepted for the interface but not sent: current Claude
        # models return a 400 when it's set.
        kwargs = {
            "modelId": self.model,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
            },
        }
        if self.effort:
            kwargs["additionalModelRequestFields"] = {
                "output_config": {"effort": self.effort}
            }
        if system_prompts:
            if cache_system:
                system_prompts.append({"cachePoint": {"type": "default"}})
            kwargs["system"] = system_prompts

        response = self.client.converse(**kwargs)

        # Models with thinking always on (Opus 5.5) put reasoningContent blocks
        # before the text, and their thinking tokens count against maxTokens.
        stop_reason = response.get("stopReason")
        if stop_reason == "max_tokens":
            raise RuntimeError(
                f"{self.model} hit maxTokens={max_tokens}; the response is truncated"
            )
        blocks = response["output"]["message"]["content"]
        texts = [b["text"] for b in blocks if "text" in b]
        if not texts:
            raise RuntimeError(
                f"{self.model} returned no text (stopReason={stop_reason}, "
                f"blocks={[next(iter(b)) for b in blocks]})"
            )
        content = "".join(texts)

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
