"""
Hypothesis execution engine for running statistical analyses.

Executes compiled hypotheses against database and returns statistical results.
"""

import pandas as pd  # type: ignore
from pydantic import BaseModel, ConfigDict, Field

from msk_cycl.hypothesis.compiler import compile_select_cohort_ids
from msk_cycl.hypothesis.db import Database
from msk_cycl.hypothesis.spec import (
    CompareCohorts,
    ComparisonMethod,
    CyclHyp,
    OverallSurvival,
    SelectCohort,
)
from msk_cycl.hypothesis.statistics import calculate_cox_hazard_ratio


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


class HypothesisExecutor:
    """Executes CYCL hypotheses against a database."""

    def __init__(self, db: Database):
        """Initialize executor with database connection."""
        self.db = db

    def execute(self, spec: CyclHyp) -> ComparisonResult:
        """Execute hypothesis and return comparison results."""
        return self._execute_compare_cohorts(spec.query)

    def _get_cohort_ids(self, cohort: SelectCohort) -> list[str]:
        """INTERNAL: Execute cohort selection and return patient IDs."""
        sql = compile_select_cohort_ids(cohort)
        df = self.db.execute(sql)
        return df["PATIENT_ID"].tolist()

    def _get_cohort_outcome_data(
        self, cohort_ids: list[str], outcome: OverallSurvival
    ) -> pd.DataFrame:
        """INTERNAL: Fetch outcome data for cohort."""
        ids_str = ", ".join(f"'{id}'" for id in cohort_ids)
        sql = f"""
            SELECT PATIENT_ID, {outcome.time_column} as time,
                   {outcome.event_column} as event
            FROM {outcome.table}
            WHERE PATIENT_ID IN ({ids_str})
        """
        return self.db.execute(sql)

    def _execute_compare_cohorts(self, query: CompareCohorts) -> ComparisonResult:
        """Execute cohort comparison using statistical method."""
        # Step 1: Get cohort IDs
        cohort_a_ids = self._get_cohort_ids(query.cohort_a)
        cohort_b_ids = self._get_cohort_ids(query.cohort_b)

        # Step 2: Get outcome data for each cohort
        cohort_a_data = self._get_cohort_outcome_data(cohort_a_ids, query.outcome)
        cohort_b_data = self._get_cohort_outcome_data(cohort_b_ids, query.outcome)

        # Step 3: Apply statistical method
        if query.method == ComparisonMethod.HAZARD_RATIO_COX:
            stats = self._calculate_cox_hazard_ratio(cohort_a_data, cohort_b_data)
        else:
            raise ValueError(f"Unsupported method: {query.method}")

        # Step 4: Return results
        return ComparisonResult(
            cohort_a_ids=cohort_a_ids,
            cohort_b_ids=cohort_b_ids,
            cohort_a_size=len(cohort_a_ids),
            cohort_b_size=len(cohort_b_ids),
            cohort_a_data=cohort_a_data,
            cohort_b_data=cohort_b_data,
            **stats,
        )

    def _calculate_cox_hazard_ratio(
        self, cohort_a_data: pd.DataFrame, cohort_b_data: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate hazard ratio using Cox proportional hazards model."""
        return calculate_cox_hazard_ratio(cohort_a_data, cohort_b_data)
