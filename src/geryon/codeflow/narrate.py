"""Narrate a code hypothesis's result via the LLM."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from geryon.codeflow.chat import (
    MessageUsage,
    parse_reply,
    reply_text,
    sum_message_usage,
)
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
        narrative = self._parse(reply_text(response, "narrator"))
        narrative.model = response.response_metadata.get("model_name")
        return narrative

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
        return parse_reply(content, CodeNarrative, "narrator")
