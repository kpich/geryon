"""AWS Bedrock provider."""

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

        session_kwargs = {}
        if profile:
            session_kwargs["profile_name"] = profile
        if region:
            session_kwargs["region_name"] = region

        session = boto3.Session(**session_kwargs)
        self.client = session.client("bedrock-runtime", region_name=self.region)

    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system_prompts = []
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
            kwargs["system"] = system_prompts

        response = self.client.converse(**kwargs)

        content = response["output"]["message"]["content"][0]["text"]

        usage = None
        if "usage" in response:
            usage = {
                "prompt_tokens": response["usage"]["inputTokens"],
                "completion_tokens": response["usage"]["outputTokens"],
                "total_tokens": response["usage"]["totalTokens"],
            }

        return LLMResponse(
            content=content,
            model=self.model_id(),
            usage=usage,
        )

    def model_id(self) -> str:
        return f"bedrock/{self.model}"
