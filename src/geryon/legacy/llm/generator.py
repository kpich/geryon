"""Hypothesis proposal model."""

from pydantic import BaseModel, Field

from geryon.legacy.lang.spec import GeryonHyp


class HypothesisProposal(BaseModel):
    """LLM-generated hypothesis proposal (pre-execution)."""

    cohort_a_description: str
    cohort_b_description: str
    outcome_description: str
    rationale: str
    geryon_spec: GeryonHyp
    refines_hypothesis: str | None = Field(
        default=None,
        description="Full hypothesis_id UUID of the parent hypothesis this refines, if "
        "any.",
    )
