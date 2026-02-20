"""Outcome definitions for hypothesis comparisons."""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field


class OverallSurvival(BaseModel):
    """Overall survival outcome definition for time-to-event analysis."""

    outcome_category: ClassVar[str] = "time_to_event"
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


class TimeToNextTreatment(BaseModel):
    """Time to next treatment (TTNT) outcome for time-to-event analysis."""

    outcome_category: ClassVar[str] = "time_to_event"
    outcome_type: Literal["time_to_next_treatment"] = "time_to_next_treatment"
    agent: str = Field(
        ...,
        description="Agent whose first administration defines the index treatment",
    )
    gap_days: int = Field(
        default=90,
        description="Treatments starting within this many days of index START_DATE "
        "are considered part of the same regimen (default 90)",
    )
    table: str = "timeline_treatment"


class SurvivalFromTreatment(BaseModel):
    """Survival measured from first administration of a specific agent."""

    outcome_category: ClassVar[str] = "time_to_event"
    outcome_type: Literal["survival_from_treatment"] = "survival_from_treatment"
    agent: str = Field(
        ..., description="Agent whose first administration defines day 0"
    )
    table: str = "timeline_treatment"


class ProgressionFromTreatment(BaseModel):
    """PFS measured from first administration of a specific agent."""

    outcome_category: ClassVar[str] = "time_to_event"
    outcome_type: Literal["progression_from_treatment"] = "progression_from_treatment"
    agent: str = Field(
        ..., description="Agent whose first administration defines day 0"
    )
    treatment_table: str = "timeline_treatment"
    progression_table: str = "timeline_progression"


class MetastaticBurden(BaseModel):
    """Number of distinct metastatic sites at a landmark timepoint."""

    outcome_category: ClassVar[str] = "continuous"
    outcome_type: Literal["metastatic_burden"] = "metastatic_burden"
    landmark_days: int = Field(
        default=0, description="Reference timepoint (days from sequencing)"
    )
    window_days: int = Field(default=30, description="Half-window for scan proximity")
    table: str = "timeline_tumor_sites"
