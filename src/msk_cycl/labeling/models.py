"""Data models for labeled hypotheses."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from msk_cycl.labeling.labels import HypothesisRating
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CyclHyp
from msk_cycl.llm.generator import HypothesisProposal
from msk_cycl.llm.narrator import HypothesisNarrative


class LabeledHypothesis(BaseModel):
    """Complete hypothesis lifecycle: proposal → execution → narrative → label.

    This model captures the full workflow from LLM generation through human review.

    Note: arbitrary_types_allowed is needed because ComparisonResult contains
    pandas DataFrames (cohort_a_data, cohort_b_data) which aren't Pydantic types.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identity
    hypothesis_id: str = Field(..., description="Unique identifier (UUID)")
    session_id: str = Field(..., description="Session/batch identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Timestamp of creation"
    )

    # Proposal phase (LLM generates hypothesis)
    proposal: HypothesisProposal = Field(..., description="LLM-generated proposal")

    # Execution phase (engine runs hypothesis)
    spec: CyclHyp = Field(..., description="Formal hypothesis specification")
    result: ComparisonResult = Field(
        ..., description="Execution results with statistics"
    )
    execution_time_seconds: float = Field(..., description="Execution duration")

    # Narration phase (LLM interprets results)
    narrative: HypothesisNarrative = Field(
        ..., description="LLM-generated interpretation"
    )
    llm_model: str = Field(..., description="Model used for generation/narration")
    iteration: int | None = Field(
        default=None, description="Iteration number within the session (1-indexed)"
    )

    # Labeling phase (human review)
    rating: HypothesisRating = Field(default_factory=HypothesisRating)
    notes: str | None = Field(
        default=None, description="Additional notes from reviewer"
    )
    labeled_at: datetime | None = Field(
        default=None, description="Timestamp of labeling"
    )
    labeled_by: str | None = Field(default=None, description="Reviewer identifier")
