"""Hypothesis generation system for MSK CYCL project."""

from msk_cycl.hypothesis.compiler import compile_hypothesis
from msk_cycl.hypothesis.db import Database
from msk_cycl.hypothesis.executor import HypothesisExecutor, HypothesisResult
from msk_cycl.hypothesis.spec import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    CyclHyp,
    OverallSurvival,
    SelectCohort,
)

__all__ = [
    # Specifications
    "CyclHyp",
    "SelectCohort",
    "CompareCohorts",
    "CohortFilter",
    "OverallSurvival",
    "ComparisonMethod",
    # Compilation
    "compile_hypothesis",
    # Execution
    "HypothesisExecutor",
    "HypothesisResult",
    # Database
    "Database",
]
