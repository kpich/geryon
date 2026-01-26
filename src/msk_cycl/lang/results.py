"""Result models for hypothesis comparisons."""

import pandas as pd  # type: ignore
from pydantic import BaseModel, ConfigDict, Field


class ComparisonResult(BaseModel):
    """Result from comparing two cohorts."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cohort_a_ids: list[str] = Field(..., description="Patient IDs in cohort A")
    cohort_b_ids: list[str] = Field(..., description="Patient IDs in cohort B")
    cohort_a_size: int = Field(..., description="Number of patients in cohort A")
    cohort_b_size: int = Field(..., description="Number of patients in cohort B")

    # Statistical results (populated by comparison method)
    hazard_ratio: float | None = None
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None
    p_value: float | None = None

    # Raw data for further analysis
    cohort_a_data: pd.DataFrame = Field(..., description="Cohort A outcome data")
    cohort_b_data: pd.DataFrame = Field(..., description="Cohort B outcome data")

    # Execution status
    success: bool = Field(default=True, description="Whether execution succeeded")
    error_message: str | None = Field(
        default=None, description="Error message if execution failed"
    )
