"""Narrate a code hypothesis's result via the LLM."""

import json

from pydantic import ValidationError

from geryon.codeflow._shared import MessageUsage, usage_from_response
from geryon.codeflow.models import CodeNarrative
from geryon.codeflow.prompts import NARRATOR_SYSTEM_PROMPT, with_focus
from geryon.llm.providers.base import ChatMessage, LLMProvider
from geryon.sandbox.result import IterationResult

MAX_STDOUT_IN_PROMPT = 2000


class CodeNarrator:
    """Generate a structured narrative from an executed script + its result."""

    def __init__(self, provider: LLMProvider, focus: str | None = None):
        self.provider = provider
        self.system_prompt = with_focus(NARRATOR_SYSTEM_PROMPT, focus)
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
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        response = self.provider.generate(messages, temperature=0.3, cache_system=True)
        self.last_usage = usage_from_response(response)
        return self._parse(response.content)

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
            return CodeNarrative(
                summary=f"Failed to parse narrative: {type(e).__name__}",
                findings=content[:500] if content else "",
                limitations=["Narrative parsing failed"],
            )
