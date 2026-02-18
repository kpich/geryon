"""
Hypothesis execution engine for running statistical analyses.

Executes compiled hypotheses against database and returns statistical results.
"""

import pandas as pd  # type: ignore

from msk_cycl.db import Database
from msk_cycl.engine.registry import get_method_implementation, get_outcome_handler
from msk_cycl.lang.compiler import (
    compile_select_cohort_ids,
    compile_select_cohort_ids_via_join,
)
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CohortFilter, CompareCohorts, CyclHyp, SelectCohort

# Tables that need a JOIN to resolve PATIENT_ID.
# Maps table -> (sample_key_column, join_table, join_column)
_PATIENT_ID_JOIN: dict[str, tuple[str, str, str]] = {
    "mutations_extended": ("Tumor_Sample_Barcode", "clinical_sample", "SAMPLE_ID"),
    "CNA": ("PATIENT_ID", "clinical_sample", "SAMPLE_ID"),
}


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
                cohort_a_size=0,
                cohort_b_size=0,
                cohort_a_data=pd.DataFrame(),
                cohort_b_data=pd.DataFrame(),
                success=False,
                error_message=str(e),
            )

    def _get_cohort_ids(self, cohort: SelectCohort) -> list[str]:
        """INTERNAL: Execute cohort selection and return patient IDs.

        When filters span multiple tables, resolves each table's filters
        independently and intersects the resulting patient ID sets.
        """
        filters_by_table: dict[str, list[CohortFilter]] = {}
        for f in cohort.filters:
            filters_by_table.setdefault(f.table, []).append(f)

        if len(filters_by_table) == 1:
            return self._resolve_single_table(cohort.filters)

        patient_id_sets: list[set[str]] = []
        for _, filters in filters_by_table.items():
            ids = self._resolve_single_table(filters)
            patient_id_sets.append(set(ids))

        result = patient_id_sets[0]
        for s in patient_id_sets[1:]:
            result &= s
        return sorted(result)

    def _resolve_single_table(self, filters: list[CohortFilter]) -> list[str]:
        """Resolve PATIENT_IDs from filters on a single table."""
        sub_cohort = SelectCohort(filters=filters)
        table = filters[0].table
        join_info = _PATIENT_ID_JOIN.get(table)

        if join_info is not None:
            sample_key, join_table, join_column = join_info
            sql = compile_select_cohort_ids_via_join(
                sub_cohort, sample_key, join_table, join_column
            )
        else:
            sql = compile_select_cohort_ids(sub_cohort)

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
            cohort_a_size=len(cohort_a_ids),
            cohort_b_size=len(cohort_b_ids),
            cohort_a_data=cohort_a_data,
            cohort_b_data=cohort_b_data,
            hazard_ratio=stats.get("hazard_ratio"),
            confidence_interval_lower=stats.get("confidence_interval_lower"),
            confidence_interval_upper=stats.get("confidence_interval_upper"),
            p_value=stats.get("p_value"),
        )
