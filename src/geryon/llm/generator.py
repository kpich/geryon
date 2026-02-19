"""Hypothesis proposal model."""

from pydantic import BaseModel

from geryon.lang.spec import GeryonHyp


class HypothesisProposal(BaseModel):
    """LLM-generated hypothesis proposal (pre-execution)."""

    cohort_a_description: str
    cohort_b_description: str
    outcome_description: str
    rationale: str
    geryon_spec: GeryonHyp
    refines_hypothesis: str | None = None
