"""
Hypothesis execution engine for running statistical analyses.

Executes compiled hypotheses against database and returns statistical results.
"""

from typing import Any, Literal

import pandas as pd  # type: ignore
from pydantic import BaseModel, ConfigDict, Field

from msk_cycl.hypothesis.db import Database
from msk_cycl.hypothesis.spec import CompareCohorts, CyclHyp, SelectCohort


class HypothesisResult(BaseModel):
    """Result from executing a hypothesis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_type: Literal["select_cohort", "compare_cohorts"]
    data: pd.DataFrame | dict[str, Any] = Field(
        ..., description="DataFrame for cohort selection, dict for comparisons"
    )


class HypothesisExecutor:
    """Executes CYCL hypotheses against a database."""

    def __init__(self, db: Database):
        """Initialize executor with database connection."""
        self.db = db

    def execute(self, spec: CyclHyp) -> HypothesisResult:
        """Execute hypothesis specification and return results."""
        if isinstance(spec.query, SelectCohort):
            return self._execute_select_cohort(spec.query)
        elif isinstance(spec.query, CompareCohorts):
            return self._execute_compare_cohorts(spec.query)
        else:
            raise ValueError(f"Unsupported query type: {type(spec.query)}")

    def _execute_select_cohort(self, query: SelectCohort) -> HypothesisResult:
        """Execute cohort selection query."""
        from msk_cycl.hypothesis.compiler import compile_select_cohort

        sql = compile_select_cohort(query)
        df = self.db.execute(sql)

        return HypothesisResult(query_type="select_cohort", data=df)

    def _execute_compare_cohorts(self, query: CompareCohorts) -> HypothesisResult:
        """Execute cohort comparison (STUB - not yet implemented)."""
        raise NotImplementedError(
            "Cohort comparison execution not yet implemented. "
            "Requires Cox regression implementation using lifelines library."
        )
