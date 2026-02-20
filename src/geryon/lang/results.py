"""Result models for hypothesis comparisons."""

import pandas as pd  # type: ignore
from pydantic import BaseModel, ConfigDict, Field


class ComparisonResult(BaseModel):
    """Result from comparing two cohorts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cohort_a_size: int = Field(..., description="Number of patients in cohort A")
    cohort_b_size: int = Field(..., description="Number of patients in cohort B")

    # Statistical results (populated by comparison method)
    hazard_ratio: float | None = None
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    p_value: float | None = None

    # Wilcoxon rank-sum results
    median_a: float | None = None
    median_b: float | None = None
    u_statistic: float | None = None

    # Method-specific stats not covered by named fields above
    extra_stats: dict[str, float] = Field(default_factory=dict)

    # Raw data for statistical calculation (not serialized to JSONL)
    cohort_a_data: pd.DataFrame | None = Field(default=None)
    cohort_b_data: pd.DataFrame | None = Field(default=None)

    # Execution status
    success: bool = Field(default=True, description="Whether execution succeeded")
    error_message: str | None = Field(
        default=None, description="Error message if execution failed"
    )
