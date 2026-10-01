from unittest.mock import MagicMock, patch

import pytest

from geryon.llm.providers.base import ChatMessage
from geryon.llm.providers.bedrock import BedrockProvider


def _provider(response: dict) -> BedrockProvider:
    with patch("geryon.llm.providers.bedrock.boto3.Session") as session:
        client = MagicMock()
        client.converse.return_value = response
        session.return_value.client.return_value = client
        return BedrockProvider(model="m")


def _response(blocks: list[dict], stop_reason: str = "end_turn") -> dict:
    return {"output": {"message": {"content": blocks}}, "stopReason": stop_reason}


MESSAGES = [ChatMessage(role="user", content="hi")]


def test_skips_reasoning_block_before_text():
    blocks = [{"reasoningContent": {"reasoningText": {"text": ""}}}, {"text": "out"}]
    assert _provider(_response(blocks)).generate(MESSAGES).content == "out"


def test_truncated_response_raises():
    with pytest.raises(RuntimeError, match="maxTokens"):
        _provider(_response([{"text": "parti"}], "max_tokens")).generate(MESSAGES)


def test_no_text_raises():
    blocks = [{"reasoningContent": {"reasoningText": {"text": ""}}}]
    with pytest.raises(RuntimeError, match="reasoningContent"):
        _provider(_response(blocks)).generate(MESSAGES)
