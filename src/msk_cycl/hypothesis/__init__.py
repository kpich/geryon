"""Hypothesis generation system for MSK CYCL project.

DEPRECATED: This module provides backward compatibility. New code should import from:
- msk_cycl.lang (hypothesis language, outcomes, methods, results)
- msk_cycl.db (database abstraction)
- msk_cycl.engine (execution engine)
"""

# Backward compatibility re-exports from new module structure
from msk_cycl.db import Database
from msk_cycl.engine import HypothesisExecutor
from msk_cycl.lang import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    ComparisonResult,
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
