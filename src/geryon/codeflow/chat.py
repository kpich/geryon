"""The chat model all three LLM phases run on, and their token-usage accounting."""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from geryon.workflow.session import SessionConfig


class MessageUsage(NamedTuple):
    """Token usage summed over one LLM phase (generation, critic, narration)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    n_llm_calls: int


def sum_message_usage(messages: list) -> MessageUsage:
    """Sum token usage across a LangGraph/LangChain message list.

    Reads each message's ``usage_metadata`` (set on AI responses). On Bedrock,
    ``input_tokens`` is the uncached input only and the ``input_token_details`` cache
    counts are separate from it, the same disjoint buckets the cost plot prices.
    """
    inp = out = tot = cache_read = cache_create = calls = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        i = int(usage.get("input_tokens", 0) or 0)
        o = int(usage.get("output_tokens", 0) or 0)
        inp += i
        out += o
        tot += int(usage.get("total_tokens", 0) or 0) or (i + o)
        details = usage.get("input_token_details") or {}
        cache_read += int(details.get("cache_read", 0) or 0)
        cache_create += int(details.get("cache_creation", 0) or 0)
        calls += 1
    return MessageUsage(inp, out, tot, cache_read, cache_create, calls)


class RefusalFallbackChat(BaseChatModel):
    """Sends every call to the primary model and re-sends a refused one to the fallback.

    Opus 5.5's safety classifiers stop some ordinary oncology text partway through
    (``stop_reason`` "refusal"). The same model tends to refuse the same content
    again, so the documented remedy is another model. Each call starts on the primary
    again; only the refused call moves. A refusal from both raises.
    """

    primary: Runnable
    fallback: Runnable
    primary_id: str
    fallback_id: str

    @property
    def _llm_type(self) -> str:
        return "refusal-fallback"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        **kwargs: Any,
    ) -> "RefusalFallbackChat":
        assert isinstance(self.primary, BaseChatModel)
        assert isinstance(self.fallback, BaseChatModel)
        return self.model_copy(
            update={
                "primary": self.primary.bind_tools(tools, **kwargs),
                "fallback": self.fallback.bind_tools(tools, **kwargs),
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self._call(self.primary, self.primary_id, messages, stop)
        if _refused(reply):
            print(
                f"⚠ {self.primary_id} refused (stop_reason=refusal); re-sending the "
                f"call to {self.fallback_id}"
            )
            reply = self._call(self.fallback, self.fallback_id, messages, stop)
            if _refused(reply):
                raise RuntimeError(
                    f"both {self.primary_id} and {self.fallback_id} refused the call "
                    f"(stop_reason=refusal)"
                )
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @staticmethod
    def _call(
        model: Runnable,
        model_id: str,
        messages: list[BaseMessage],
        stop: list[str] | None,
    ) -> AIMessage:
        reply = (
            model.invoke(_for_model(messages, model_id), stop=stop)
            if stop
            else model.invoke(_for_model(messages, model_id))
        )
        assert isinstance(reply, AIMessage)
        return reply


def _refused(reply: AIMessage) -> bool:
    return reply.additional_kwargs.get("stop_reason") == "refusal"


def _for_model(messages: list[BaseMessage], model_id: str) -> list[BaseMessage]:
    """Drop thinking blocks that another model wrote.

    Thinking blocks are signed for the model that produced them. After a fallback the
    history holds both models' turns, and neither can use the other's blocks.
    """
    out: list[BaseMessage] = []
    for msg in messages:
        author = msg.response_metadata.get("model_name")
        if (
            isinstance(msg, AIMessage)
            and isinstance(msg.content, list)
            and author is not None
            and author != model_id
        ):
            content = [
                b
                for b in msg.content
                if not (
                    isinstance(b, dict)
                    and b.get("type") in ("thinking", "redacted_thinking")
                )
            ]
            msg = msg.model_copy(update={"content": content})
        out.append(msg)
    return out


def _bedrock(config: SessionConfig, model_id: str) -> ChatBedrock:
    # Bedrock 503s come in bursts; legacy mode's 2 quick retries let one burst end a
    # multi-hour session. Adaptive backs off exponentially and still raises in the end.
    boto_config = BotoConfig(
        read_timeout=300,
        connect_timeout=30,
        retries={"mode": "adaptive", "max_attempts": 8},
    )
    kwargs: dict = {
        "model_id": model_id,
        # Current Claude models 400 on temperature/top_p/top_k, so none are sent.
        "model_kwargs": {
            "max_tokens": 16384,
            "output_config": {"effort": config.effort},
        },
        "config": boto_config,
    }
    if config.aws_region:
        kwargs["region_name"] = config.aws_region
    if config.aws_profile:
        kwargs["credentials_profile_name"] = config.aws_profile
    if "arn:" in model_id or "anthropic" in model_id.lower():
        kwargs["provider"] = "anthropic"
    return ChatBedrock(**kwargs)


def build_chat_model(config: SessionConfig) -> RefusalFallbackChat:
    """Build the chat model all three roles run on, with its refusal fallback."""
    return RefusalFallbackChat(
        primary=_bedrock(config, config.model),
        fallback=_bedrock(config, config.fallback_model),
        primary_id=config.model,
        fallback_id=config.fallback_model,
    )
