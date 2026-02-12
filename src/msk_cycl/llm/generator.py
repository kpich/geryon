"""Hypothesis proposal model."""

from pydantic import BaseModel

from msk_cycl.lang.spec import CyclHyp


class HypothesisProposal(BaseModel):
    """LLM-generated hypothesis proposal (pre-execution)."""

    cohort_a_description: str
    cohort_b_description: str
    outcome_description: str
    rationale: str
    cycl_spec: CyclHyp
