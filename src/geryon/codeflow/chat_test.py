from pathlib import Path
from unittest.mock import patch

from geryon.codeflow.chat import build_chat_model
from geryon.workflow.session import SessionConfig


def _config(**kw) -> SessionConfig:
    return SessionConfig(parquet_dir=Path("p"), storage_dir=Path("s"), **kw)


def test_bedrock_chat_model_sends_effort():
    with patch("geryon.codeflow.chat.ChatBedrock") as chat:
        build_chat_model(_config(effort="xhigh"))
    model_kwargs = chat.call_args.kwargs["model_kwargs"]
    assert model_kwargs["output_config"] == {"effort": "xhigh"}


def test_effort_defaults_to_medium():
    assert _config().effort == "medium"
