"""Hypothesis proposal generation using LLM."""

import json

from pydantic import BaseModel

from msk_cycl.lang.spec import CyclHyp
from msk_cycl.llm.prompts import GENERATOR_SYSTEM_PROMPT
from msk_cycl.llm.providers.base import ChatMessage, LLMProvider
from msk_cycl.llm.schema import DatabaseSchema, schema_to_context


class HypothesisProposal(BaseModel):
    """LLM-generated hypothesis proposal (pre-execution)."""

    cohort_a_description: str
    cohort_b_description: str
    outcome_description: str
    rationale: str
    cycl_spec: CyclHyp


class HypothesisGenerator:
    """Generate hypothesis proposals using LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        schema: DatabaseSchema,
        previous_hypotheses: list | None = None,
    ):
        """Initialize generator.

        Parameters
        ----------
        provider : LLMProvider
            LLM provider for generation
        schema : DatabaseSchema
            Database schema for context
        previous_hypotheses : list, optional
            Previously generated hypotheses to avoid duplicates
        """
        self.provider = provider
        self.schema = schema
        self.previous_hypotheses = previous_hypotheses or []

    def propose(self, n: int = 1) -> list[HypothesisProposal]:
        """Generate N hypothesis proposals.

        Parameters
        ----------
        n : int
            Number of proposals to generate

        Returns
        -------
        list[HypothesisProposal]
            Generated proposals
        """
        system_prompt = self._build_system_prompt()
        user_prompt = (
            f"Propose {n} novel hypothesis(es) for cancer cohort comparison. "
            "Return ONLY valid JSON matching the schema above."
        )

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = self.provider.generate(messages, temperature=0.8)

        # Parse JSON response
        proposals = self._parse_proposals(response.content)
        return proposals

    def _build_system_prompt(self) -> str:
        """Construct system prompt with schema and instructions."""
        schema_context = schema_to_context(self.schema)
        previous_context = self._format_previous_hypotheses()

        return GENERATOR_SYSTEM_PROMPT.format(
            schema=schema_context,
            previous_hypotheses=previous_context,
        )

    def _format_previous_hypotheses(self) -> str:
        """Format previous hypotheses for context."""
        if not self.previous_hypotheses:
            return "No previous hypotheses yet."

        lines = ["Previously tested hypotheses (avoid duplicates):"]
        for i, hyp in enumerate(self.previous_hypotheses[:10], 1):  # Show max 10
            # Extract description if it's a LabeledHypothesis
            if hasattr(hyp, "proposal"):
                a_desc = hyp.proposal.cohort_a_description
                b_desc = hyp.proposal.cohort_b_description
                desc = f"{a_desc} vs {b_desc}"
            else:
                desc = str(hyp)
            lines.append(f"{i}. {desc}")

        if len(self.previous_hypotheses) > 10:
            lines.append(f"... and {len(self.previous_hypotheses) - 10} more")

        return "\n".join(lines)

    def _parse_proposals(self, content: str) -> list[HypothesisProposal]:
        """Parse LLM JSON response into proposals.

        Parameters
        ----------
        content : str
            LLM response content

        Returns
        -------
        list[HypothesisProposal]
            Parsed proposals
        """
        # Strip whitespace
        content = content.strip()

        # With format="json", Ollama returns pure JSON (no markdown)
        # But keep markdown handling as fallback for other providers
        if content.startswith("```"):
            # Remove markdown code blocks
            lines = content.split("\n")

            # Find the lines with ``` markers
            code_block_markers = []
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    code_block_markers.append(i)

            # Extract content between first and last markers
            if len(code_block_markers) >= 2:
                start_idx = code_block_markers[0] + 1
                end_idx = code_block_markers[-1]
                content = "\n".join(lines[start_idx:end_idx])
            elif len(code_block_markers) == 1:
                # Only opening marker, take everything after it
                start_idx = code_block_markers[0] + 1
                content = "\n".join(lines[start_idx:])

        content = content.strip()

        # Add better error handling
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Print diagnostic info
            print(f"ERROR: Failed to parse JSON: {e}")
            print("Content (first 500 chars):")
            print("=" * 80)
            print(content[:500])
            print("=" * 80)
            raise ValueError(
                f"Failed to parse LLM response as JSON. "
                f"Error: {e}. Content starts with: {content[:100]}"
            ) from e

        # Handle both {"proposals": [...]} and direct array
        if isinstance(data, dict) and "proposals" in data:
            proposals_data = data["proposals"]
        elif isinstance(data, list):
            proposals_data = data
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

        # Parse each proposal
        proposals = []
        for prop_data in proposals_data:
            proposal = HypothesisProposal(**prop_data)
            proposals.append(proposal)

        return proposals
