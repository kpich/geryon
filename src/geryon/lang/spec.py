"""
Pydantic models for Geryon hypothesis specifications.

Defines type-safe structures for expressing cohort selection queries.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from geryon.lang.methods import ComparisonMethod
from geryon.lang.outcomes import (
    MetastaticBurden,
    OverallSurvival,
    SurvivalFromTreatment,
    TimeToNextTreatment,
)

# Query language version for backward compatibility tracking
QUERY_LANGUAGE_VERSION = 1

Outcome = Annotated[
    OverallSurvival | TimeToNextTreatment | SurvivalFromTreatment | MetastaticBurden,
    Field(discriminator="outcome_type"),
]

_TIME_TO_EVENT = {
    "overall_survival",
    "time_to_next_treatment",
    "survival_from_treatment",
}
_CONTINUOUS = {"metastatic_burden"}


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
    """
    Select patients matching filter criteria.

    INTERNAL USE ONLY: This class is used internally by CompareCohorts
    to define cohorts. Users should not create GeryonHyp objects with
    bare SelectCohort queries - use CompareCohorts instead.
    """

    operation: Literal["select_cohort"] = "select_cohort"
    filters: list[CohortFilter] = Field(..., description="Filter criteria (ANDed)")


class CompareCohorts(BaseModel):
    """Compare two cohorts using statistical analysis."""

    operation: Literal["compare_cohorts"] = "compare_cohorts"
    cohort_a: SelectCohort = Field(..., description="First cohort (treatment/exposure)")
    cohort_b: SelectCohort = Field(..., description="Second cohort (control/reference)")
    outcome: Outcome = Field(..., description="Outcome to compare")
    method: ComparisonMethod = Field(..., description="Statistical comparison method")

    @model_validator(mode="after")
    def validate_method_outcome_compatibility(self) -> "CompareCohorts":
        """Validate that the comparison method is compatible with the outcome type."""
        otype = self.outcome.outcome_type
        if (
            self.method == ComparisonMethod.HAZARD_RATIO_COX
            and otype not in _TIME_TO_EVENT
        ):
            raise ValueError(
                f"hazard_ratio_cox requires a time-to-event outcome "
                f"({', '.join(sorted(_TIME_TO_EVENT))}), got {otype}"
            )
        if (
            self.method == ComparisonMethod.WILCOXON_RANK_SUM
            and otype not in _CONTINUOUS
        ):
            raise ValueError(
                f"wilcoxon_rank_sum requires a continuous outcome "
                f"({', '.join(sorted(_CONTINUOUS))}), got {otype}"
            )
        return self


class GeryonHyp(BaseModel):
    """Top-level Geryon hypothesis specification."""

    version: Literal[1] = Field(
        default=1, description="Query language version for compatibility"
    )
    query: CompareCohorts
