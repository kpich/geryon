"""
Pydantic models for CYCL hypothesis specifications.

Defines type-safe structures for expressing cohort selection queries.
"""

from typing import Literal

from pydantic import BaseModel, Field

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


class CyclHyp(BaseModel):
    """Top-level CYCL hypothesis specification."""

    version: Literal[1] = Field(
        default=1, description="Query language version for compatibility"
    )
    query: SelectCohort

    # Future: add comparison, outcome, adjustment, etc.
