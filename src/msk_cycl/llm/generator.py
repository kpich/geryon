"""Hypothesis proposal generation using LLM."""

import instructor
from openai import OpenAI
from pydantic import BaseModel

from msk_cycl.lang.spec import CyclHyp
from msk_cycl.llm.conversation_logger import ConversationLogger
from msk_cycl.llm.prompts import GENERATOR_SYSTEM_PROMPT
from msk_cycl.llm.providers.base import LLMProvider
from msk_cycl.llm.schema import DatabaseSchema, schema_to_context


class HypothesisProposal(BaseModel):
    """LLM-generated hypothesis proposal (pre-execution)."""

    cohort_a_description: str
    cohort_b_description: str
    outcome_description: str
    rationale: str
    cycl_spec: CyclHyp


class ProposalsList(BaseModel):
    """Wrapper for list of proposals."""

    proposals: list[HypothesisProposal]


class HypothesisGenerator:
    """Generate hypothesis proposals using LLM."""

    def __init__(
        self,
        provider: LLMProvider,
        schema: DatabaseSchema,
        previous_hypotheses: list | None = None,
        logger: ConversationLogger | None = None,
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
        logger : ConversationLogger, optional
            Logger for LLM conversations
        """
        self.provider = provider
        self.schema = schema
        self.previous_hypotheses = previous_hypotheses or []
        self.logger = logger

        # Set up Instructor client for structured output
        # Ollama is OpenAI-compatible via /v1 endpoint
        base_url = getattr(provider, "base_url", "http://localhost:11434")
        self.model = getattr(provider, "model", "mixtral:8x7b")
        self.client = instructor.from_openai(
            OpenAI(
                base_url=f"{base_url}/v1",
                api_key="ollama",  # Dummy key for Ollama
            ),
            mode=instructor.Mode.JSON,
        )

    def propose(self, n: int = 1) -> list[HypothesisProposal]:
        """Generate N hypothesis proposals.

        Parameters
        ----------
        n : int
            Number of proposals to generate

        Returns
        -------
        list[HypothesisProposal]
            Generated proposals (may be empty if LLM fails validation)
        """
        system_prompt = self._build_system_prompt()
        user_prompt = f"Propose {n} novel hypothesis(es) for cancer cohort comparison."

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use Instructor for structured output
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_model=ProposalsList,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.8,
            )

            # Log successful interaction
            if self.logger:
                self.logger.log_interaction(
                    interaction_type="hypothesis_generation",
                    messages=messages,
                    response={
                        "content": response.model_dump_json(indent=2),
                        "model": self.provider.model_id(),
                    },
                    usage=None,  # Instructor doesn't expose usage stats easily
                    metadata={"temperature": 0.8, "status": "success"},
                )

            return response.proposals

        except Exception as e:
            # Log failed interaction
            if self.logger:
                self.logger.log_interaction(
                    interaction_type="hypothesis_generation",
                    messages=messages,
                    response={
                        "content": f"ERROR: {type(e).__name__}: {str(e)}",
                        "model": self.provider.model_id(),
                    },
                    usage=None,
                    metadata={"temperature": 0.8, "status": "error"},
                )

            # Log to console for visibility
            print("⚠ WARNING: Hypothesis generation failed after retries")
            print(f"  Error: {type(e).__name__}")
            print("  Details logged to LLM chat file")

            # Return empty list - don't crash the workflow
            return []

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
