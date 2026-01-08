"""CYCL hypothesis language - specifications, outcomes, methods, and results."""

from msk_cycl.lang.compiler import (
    compile_cohort_filter,
    compile_select_cohort,
    compile_select_cohort_ids,
)
from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import (
    CohortFilter,
    CompareCohorts,
    CyclHyp,
)
from msk_cycl.lang.spec import (
    SelectCohort as SelectCohort,  # Available for internal use (not in __all__)
)

__all__ = [
    # Hypothesis specification
    "CyclHyp",
    "CompareCohorts",
    "CohortFilter",
    # Outcomes
    "OverallSurvival",
    # Comparison methods
    "ComparisonMethod",
    # Results
    "ComparisonResult",
    # SQL compilation
    "compile_cohort_filter",
    "compile_select_cohort",
    "compile_select_cohort_ids",
]

# NOTE: SelectCohort is available for import but not in __all__
# It's internal and should not be used directly by users
