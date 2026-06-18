"""
Hypothesis execution engine for running statistical analyses.

Executes compiled hypotheses against database and returns statistical results.
"""

import pandas as pd  # type: ignore

from geryon.db import Database
from geryon.engine.registry import (
    TABLE_PATIENT_ID_JOINS,
    get_method_implementation,
    get_outcome_handler,
)
from geryon.lang.compiler import (
    compile_select_cohort_ids,
    compile_select_cohort_ids_via_join,
)
from geryon.lang.methods import ComparisonMethod
from geryon.lang.results import ComparisonResult
from geryon.lang.spec import (
    CohortFilter,
    CompareCohorts,
    GeryonHyp,
    Outcome,
    SelectCohort,
)

_KNOWN_STAT_KEYS = frozenset(
    {
        "hazard_ratio",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "p_value",
        "median_a",
        "median_b",
        "u_statistic",
    }
)


def load_split_ids(db: Database, split: str) -> frozenset[str] | None:
    """Return patient IDs for a named split, or None if no split table exists."""
    if "patient_split" not in db.list_tables():
        return None
    df = db.execute(
        f'SELECT "PATIENT_ID" FROM "patient_split" WHERE "split" = \'{split}\''
    )
    return frozenset(df["PATIENT_ID"].tolist())


class HypothesisExecutor:
    """Executes Geryon hypotheses against a database."""

    def __init__(self, db: Database, patient_ids: frozenset[str] | None = None):
        self.db = db
        self._patient_ids = patient_ids

    def execute(self, spec: GeryonHyp) -> ComparisonResult:
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
            ids = self._resolve_single_table(cohort.filters)
            if self._patient_ids is not None:
                ids = [pid for pid in ids if pid in self._patient_ids]
            return ids

        patient_id_sets: list[set[str]] = []
        for _, filters in filters_by_table.items():
            ids = self._resolve_single_table(filters)
            patient_id_sets.append(set(ids))

        result = patient_id_sets[0]
        for s in patient_id_sets[1:]:
            result &= s
        ids = sorted(result)
        if self._patient_ids is not None:
            ids = [pid for pid in ids if pid in self._patient_ids]
        return ids

    def _resolve_single_table(self, filters: list[CohortFilter]) -> list[str]:
        """Resolve PATIENT_IDs from filters on a single table."""
        sub_cohort = SelectCohort(filters=filters)
        table = filters[0].table
        join_info = TABLE_PATIENT_ID_JOINS.get(table)

        if join_info is not None:
            sample_key, join_table, join_column = join_info
            sql = compile_select_cohort_ids_via_join(
                sub_cohort, sample_key, join_table, join_column
            )
        else:
            sql = compile_select_cohort_ids(sub_cohort)

        df = self.db.execute(sql)
        return df["PATIENT_ID"].tolist()

    def compare_ids(
        self,
        cohort_a_ids: list[str],
        cohort_b_ids: list[str],
        outcome: Outcome,
        method: ComparisonMethod,
    ) -> ComparisonResult:
        """Compare pre-resolved patient ID lists without cohort-ID resolution."""
        try:
            outcome_handler = get_outcome_handler(outcome)
            cohort_a_data = outcome_handler.extract_data(cohort_a_ids, outcome, self.db)
            cohort_b_data = outcome_handler.extract_data(cohort_b_ids, outcome, self.db)
            method_impl = get_method_implementation(method)
            stats = method_impl.calculate(cohort_a_data, cohort_b_data)
            return ComparisonResult(
                cohort_a_size=len(cohort_a_data),
                cohort_b_size=len(cohort_b_data),
                cohort_a_data=cohort_a_data,
                cohort_b_data=cohort_b_data,
                **{k: v for k, v in stats.items() if k in _KNOWN_STAT_KEYS},  # type: ignore[arg-type]
                extra_stats={
                    k: v for k, v in stats.items() if k not in _KNOWN_STAT_KEYS
                },
            )
        except Exception as e:
            return ComparisonResult(
                cohort_a_size=0,
                cohort_b_size=0,
                success=False,
                error_message=str(e),
            )

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
            **{k: v for k, v in stats.items() if k in _KNOWN_STAT_KEYS},  # type: ignore[arg-type]
            extra_stats={k: v for k, v in stats.items() if k not in _KNOWN_STAT_KEYS},
        )
