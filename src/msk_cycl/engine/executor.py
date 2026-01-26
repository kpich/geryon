"""
Hypothesis execution engine for running statistical analyses.

Executes compiled hypotheses against database and returns statistical results.
"""

import pandas as pd  # type: ignore

from msk_cycl.db import Database
from msk_cycl.engine.registry import get_method_implementation, get_outcome_handler
from msk_cycl.lang.compiler import compile_select_cohort_ids
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CompareCohorts, CyclHyp, SelectCohort


class HypothesisExecutor:
    """Executes CYCL hypotheses against a database."""

    def __init__(self, db: Database):
        """Initialize executor with database connection."""
        self.db = db

    def execute(self, spec: CyclHyp) -> ComparisonResult:
        """Execute hypothesis and return comparison results."""
        try:
            return self._execute_compare_cohorts(spec.query)
        except Exception as e:
            return ComparisonResult(
                cohort_a_ids=[],
                cohort_b_ids=[],
                cohort_a_size=0,
                cohort_b_size=0,
                cohort_a_data=pd.DataFrame(),
                cohort_b_data=pd.DataFrame(),
                success=False,
                error_message=str(e),
            )

    def _get_cohort_ids(self, cohort: SelectCohort) -> list[str]:
        """INTERNAL: Execute cohort selection and return patient IDs."""
        sql = compile_select_cohort_ids(cohort)
        df = self.db.execute(sql)
        return df["PATIENT_ID"].tolist()

    def _execute_compare_cohorts(self, query: CompareCohorts) -> ComparisonResult:
        """Execute cohort comparison using statistical method.

        This method uses the registry pattern to look up:
        1. Outcome handler (based on outcome type)
        2. Comparison method implementation (based on method enum)

        This makes the code declarative and extensible - no if/elif chains!
        """
        cohort_a_ids = self._get_cohort_ids(query.cohort_a)
        cohort_b_ids = self._get_cohort_ids(query.cohort_b)

        outcome_handler = get_outcome_handler(query.outcome)
        cohort_a_data = outcome_handler.extract_data(
            cohort_a_ids, query.outcome, self.db
        )
        cohort_b_data = outcome_handler.extract_data(
            cohort_b_ids, query.outcome, self.db
        )

        method_impl = get_method_implementation(query.method)
        stats = method_impl.calculate(cohort_a_data, cohort_b_data)

        return ComparisonResult(
            cohort_a_ids=cohort_a_ids,
            cohort_b_ids=cohort_b_ids,
            cohort_a_size=len(cohort_a_ids),
            cohort_b_size=len(cohort_b_ids),
            cohort_a_data=cohort_a_data,
            cohort_b_data=cohort_b_data,
            hazard_ratio=stats.get("hazard_ratio"),
            confidence_interval_lower=stats.get("confidence_interval_lower"),
            confidence_interval_upper=stats.get("confidence_interval_upper"),
            p_value=stats.get("p_value"),
        )
