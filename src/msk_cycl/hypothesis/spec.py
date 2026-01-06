"""
Pydantic models for CYCL hypothesis specifications.

Defines type-safe structures for expressing cohort selection queries.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Query language version for backward compatibility tracking
QUERY_LANGUAGE_VERSION = 1


class CohortFilter(BaseModel):
    """Single filter criterion for cohort selection."""

    table: str = Field(..., description="Table name (e.g., 'clinical_patient')")
    column: str = Field(..., description="Column name to filter on")
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "in"] = Field(
        ..., description="Comparison operator"
    )
    value: str | int | float | list[str | int | float] = Field(
        ..., description="Value(s) to compare against"
    )


class SelectCohort(BaseModel):
    """Select patients matching filter criteria."""

    operation: Literal["select_cohort"] = "select_cohort"
    filters: list[CohortFilter] = Field(..., description="Filter criteria (ANDed)")


class OverallSurvival(BaseModel):
    """Overall survival outcome definition for time-to-event analysis."""

    outcome_type: Literal["overall_survival"] = "overall_survival"
    time_column: str = Field(
        default="OS_MONTHS",
        description="Column containing survival time (cBioPortal default: OS_MONTHS)",
    )
    event_column: str = Field(
        default="OS_STATUS",
        description="Column containing event indicator (1=death, 0=censored)",
    )
    table: str = Field(
        default="clinical_patient",
        description="Table containing outcome data (typically same as cohort table)",
    )


class ComparisonMethod(str, Enum):
    """Statistical method for comparing cohorts."""

    HAZARD_RATIO_COX = "hazard_ratio_cox"


class CompareCohorts(BaseModel):
    """Compare two cohorts using statistical analysis."""

    operation: Literal["compare_cohorts"] = "compare_cohorts"
    cohort_a: SelectCohort = Field(..., description="First cohort (treatment/exposure)")
    cohort_b: SelectCohort = Field(..., description="Second cohort (control/reference)")
    outcome: OverallSurvival = Field(..., description="Outcome to compare")
    method: ComparisonMethod = Field(..., description="Statistical comparison method")

    @model_validator(mode="after")
    def validate_method_outcome_compatibility(self) -> "CompareCohorts":
        """Validate that the comparison method is compatible with the outcome type."""
        if (
            self.method == ComparisonMethod.HAZARD_RATIO_COX
            and self.outcome.outcome_type != "overall_survival"
        ):
            raise ValueError(
                "hazard_ratio_cox requires time-to-event outcome "
                f"(overall_survival), got {self.outcome.outcome_type}"
            )
        return self


class CyclHyp(BaseModel):
    """Top-level CYCL hypothesis specification."""

    version: Literal[1] = Field(
        default=1, description="Query language version for compatibility"
    )
    query: SelectCohort | CompareCohorts
