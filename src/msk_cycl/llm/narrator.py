"""Result narration - LLM generates human-readable interpretations."""

import json

from pydantic import BaseModel

from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CyclHyp
from msk_cycl.llm.prompts import NARRATOR_SYSTEM_PROMPT
from msk_cycl.llm.providers.base import ChatMessage, LLMProvider


class HypothesisNarrative(BaseModel):
    """LLM-generated narrative of results."""

    summary: str
    findings: str
    limitations: list[str]
    clinical_relevance: str


class ResultNarrator:
    """Generate human-readable narratives from ComparisonResult."""

    def __init__(self, provider: LLMProvider):
        """Initialize narrator.

        Parameters
        ----------
        provider : LLMProvider
            LLM provider for narrative generation
        """
        self.provider = provider

    def narrate(
        self,
        spec: CyclHyp,
        result: ComparisonResult,
        proposal_rationale: str,
    ) -> HypothesisNarrative:
        """Generate narrative from execution results.

        Parameters
        ----------
        spec : CyclHyp
            Original hypothesis specification
        result : ComparisonResult
            Execution results with statistics
        proposal_rationale : str
            Original rationale from proposal

        Returns
        -------
        HypothesisNarrative
            Generated narrative
        """
        system_prompt = NARRATOR_SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(spec, result, proposal_rationale)

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = self.provider.generate(messages, temperature=0.3)

        # Parse JSON response
        narrative = self._parse_narrative(response.content)
        return narrative

    def _build_user_prompt(
        self,
        spec: CyclHyp,
        result: ComparisonResult,
        proposal_rationale: str,
    ) -> str:
        """Build user prompt with hypothesis and results."""
        # Extract comparison details
        query = spec.query
        outcome = query.outcome

        # Format results
        stats_lines = [
            "# STATISTICAL RESULTS",
            "",
            f"**Cohort A**: {result.cohort_a_size} patients",
            f"**Cohort B**: {result.cohort_b_size} patients",
            "",
        ]

        if result.hazard_ratio is not None:
            ci_lower = result.confidence_interval_lower
            ci_upper = result.confidence_interval_upper
            stats_lines.extend(
                [
                    f"**Hazard Ratio (HR)**: {result.hazard_ratio:.3f}",
                    f"**95% Confidence Interval**: [{ci_lower:.3f}, {ci_upper:.3f}]",
                    f"**P-value**: {result.p_value:.4f}",
                ]
            )

        prompt = f"""# HYPOTHESIS

**Original Rationale**: {proposal_rationale}

**Outcome**: {outcome.outcome_type}

{chr(10).join(stats_lines)}

# TASK

Interpret these results and provide a structured narrative in JSON format
as specified in your instructions. Return ONLY valid JSON.
"""

        return prompt

    def _parse_narrative(self, content: str) -> HypothesisNarrative:
        """Parse LLM JSON response into narrative.

        Parameters
        ----------
        content : str
            LLM response content

        Returns
        -------
        HypothesisNarrative
            Parsed narrative
        """
        # Extract JSON from content (may be wrapped in markdown)
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code blocks
            lines = content.split("\n")
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start_idx == 0:
                        start_idx = i + 1
                    else:
                        end_idx = i
                        break
            content = "\n".join(lines[start_idx:end_idx])

        # Parse JSON
        data = json.loads(content)
        return HypothesisNarrative(**data)
