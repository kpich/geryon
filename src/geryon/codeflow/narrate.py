"""Narrate a code hypothesis's result via the LLM."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from geryon.codeflow.chat import MessageUsage, sum_message_usage
from geryon.codeflow.models import CodeNarrative
from geryon.codeflow.prompts import with_data_dictionary, with_focus
from geryon.sandbox.result import IterationResult

MAX_STDOUT_IN_PROMPT = 2000


class CodeNarrator:
    """Generate a structured narrative from an executed script + its result."""

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        focus: str | None = None,
        data_dictionary: str | None = None,
    ):
        self.llm = llm
        self.system_prompt = with_focus(
            with_data_dictionary(system_prompt, data_dictionary), focus
        )
        # Usage of the most recent narrate() call, for cost logging.
        self.last_usage: MessageUsage | None = None

    def narrate(
        self,
        *,
        description: str,
        rationale: str,
        code: str,
        result: IterationResult | None,
        stdout: str,
    ) -> CodeNarrative:
        user_prompt = self._build_user_prompt(
            description=description,
            rationale=rationale,
            code=code,
            result=result,
            stdout=stdout,
        )
        # No cache breakpoint: ChatBedrock's InvokeModel path flattens a block-list
        # system prompt to a string, dropping it, and the narrator's prompt never
        # reached the cache minimum anyway (zero cache reads before this path).
        response = self.llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        self.last_usage = sum_message_usage([response])
        return self._parse(_response_text(response))

    def _build_user_prompt(
        self,
        *,
        description: str,
        rationale: str,
        code: str,
        result: IterationResult | None,
        stdout: str,
    ) -> str:
        result_block = (
            json.dumps(result.model_dump(), indent=2, default=str)
            if result is not None
            else "(the script did not report a standardized result)"
        )
        stdout_tail = stdout[-MAX_STDOUT_IN_PROMPT:] if stdout else "(no stdout)"
        return f"""# HYPOTHESIS

{description}

**Rationale**: {rationale}

# CODE

```python
{code}
```

# REPORTED RESULT

{result_block}

# STDOUT (tail)

{stdout_tail}

# TASK

Interpret this result. Return ONLY valid JSON with keys summary, findings,
limitations, context_summary.
"""

    def _parse(self, content: str) -> CodeNarrative:
        try:
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                start, end = 0, len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        if start == 0:
                            start = i + 1
                        else:
                            end = i
                            break
                text = "\n".join(lines[start:end])
            return CodeNarrative(**json.loads(text))
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(
                f"narrator returned unparseable output ({type(e).__name__}): "
                f"{content[:500]!r}"
            ) from e


def _response_text(response: AIMessage) -> str:
    """Text of a narrator reply, raising if it was truncated or held no text.

    With thinking on (Opus 5.5 always), the content is a block list with thinking
    blocks before the text, and thinking tokens count against ``max_tokens``.
    """
    stop_reason = response.additional_kwargs.get("stop_reason")
    if stop_reason == "max_tokens":
        raise RuntimeError("narrator hit max_tokens; the response is truncated")
    content = response.content
    block_types: list[str | None]
    if isinstance(content, str):
        text, block_types = content, ["text"]
    else:
        blocks = [b for b in content if isinstance(b, dict)]
        text = "".join(b["text"] for b in blocks if b.get("type") == "text")
        block_types = [b.get("type") for b in blocks]
    if not text.strip():
        raise RuntimeError(
            f"narrator returned no text (stop_reason={stop_reason}, "
            f"blocks={block_types})"
        )
    return text
