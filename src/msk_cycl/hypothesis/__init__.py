"""Hypothesis generation system for MSK CYCL project."""

from msk_cycl.hypothesis.db import Database
from msk_cycl.hypothesis.executor import ComparisonResult, HypothesisExecutor
from msk_cycl.hypothesis.spec import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    CyclHyp,
    OverallSurvival,
)

__all__ = [
    # Public API - Hypothesis Definition
    "CyclHyp",
    "CompareCohorts",
    "CohortFilter",
    "OverallSurvival",
    "ComparisonMethod",
    # Public API - Execution
    "HypothesisExecutor",
    "ComparisonResult",
    # Public API - Database
    "Database",
]

# NOTE: SelectCohort is intentionally NOT exported
# It is used internally by CompareCohorts but should not be
# constructed directly by users as a hypothesis
