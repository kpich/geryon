"""Outcome definitions for hypothesis comparisons."""

from typing import Literal

from pydantic import BaseModel, Field


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
