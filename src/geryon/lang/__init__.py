"""Geryon hypothesis language - specifications, outcomes, methods, and results."""

from geryon.lang.compiler import (
    compile_cohort_filter,
    compile_select_cohort,
    compile_select_cohort_ids,
)
from geryon.lang.methods import ComparisonMethod
from geryon.lang.outcomes import (
    MetastaticBurden,
    OverallSurvival,
    ProgressionFromTreatment,
    SurvivalFromTreatment,
    TimeToNextTreatment,
)
from geryon.lang.results import ComparisonResult
from geryon.lang.spec import (
    CohortFilter,
    CompareCohorts,
    GeryonHyp,
)
from geryon.lang.spec import (
    SelectCohort as SelectCohort,  # Available for internal use (not in __all__)
)

__all__ = [
    # Hypothesis specification
    "GeryonHyp",
    "CompareCohorts",
    "CohortFilter",
    # Outcomes
    "OverallSurvival",
    "TimeToNextTreatment",
    "SurvivalFromTreatment",
    "ProgressionFromTreatment",
    "MetastaticBurden",
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
