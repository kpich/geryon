"""Registry for outcome handlers and comparison method implementations.

This registry provides a declarative mapping between:
- Outcome types → Handler classes
- Comparison methods → Implementation classes

Adding new outcomes or methods is as simple as:
1. Creating the handler/implementation class
2. Adding an entry to the registry below
"""

from typing import Any

from geryon.engine.methods import (
    ComparisonMethodImpl,
    CoxHazardRatioMethod,
    WilcoxonRankSumMethod,
)
from geryon.engine.outcomes import (
    MetastaticBurdenHandler,
    OutcomeHandler,
    OverallSurvivalHandler,
    SurvivalFromTreatmentHandler,
    TimeToNextTreatmentHandler,
)
from geryon.lang.methods import ComparisonMethod
from geryon.lang.outcomes import (
    MetastaticBurden,
    OverallSurvival,
    SurvivalFromTreatment,
    TimeToNextTreatment,
)

# ============================================================================
# OUTCOME HANDLERS REGISTRY
# ============================================================================
# Maps outcome type → handler class
# To add a new outcome: create handler class and add mapping here

OUTCOME_HANDLERS: dict[type[Any], type[Any]] = {
    OverallSurvival: OverallSurvivalHandler,
    TimeToNextTreatment: TimeToNextTreatmentHandler,
    SurvivalFromTreatment: SurvivalFromTreatmentHandler,
    MetastaticBurden: MetastaticBurdenHandler,
}

# ============================================================================
# COMPARISON METHOD IMPLEMENTATIONS REGISTRY
# ============================================================================
# Maps comparison method enum → implementation class
# To add a new method: create implementation class and add mapping here

METHOD_IMPLEMENTATIONS: dict[ComparisonMethod, type[Any]] = {
    ComparisonMethod.HAZARD_RATIO_COX: CoxHazardRatioMethod,
    ComparisonMethod.WILCOXON_RANK_SUM: WilcoxonRankSumMethod,
}


# ============================================================================
# REGISTRY ACCESSOR FUNCTIONS
# ============================================================================


def get_outcome_handler(outcome: object) -> OutcomeHandler:
    """Get handler instance for an outcome.

    Parameters
    ----------
    outcome : object
        Outcome specification (e.g., OverallSurvival instance)

    Returns
    -------
    OutcomeHandler
        Handler instance for this outcome type

    Raises
    ------
    ValueError
        If outcome type is not registered
    """
    outcome_type = type(outcome)
    handler_class = OUTCOME_HANDLERS.get(outcome_type)

    if handler_class is None:
        registered = ", ".join(cls.__name__ for cls in OUTCOME_HANDLERS)
        raise ValueError(
            f"Unsupported outcome type: {outcome_type.__name__}. "
            f"Registered outcomes: {registered}"
        )

    return handler_class()


def get_method_implementation(method: ComparisonMethod) -> ComparisonMethodImpl:
    """Get implementation instance for a comparison method.

    Parameters
    ----------
    method : ComparisonMethod
        Comparison method enum value

    Returns
    -------
    ComparisonMethodImpl
        Implementation instance for this method

    Raises
    ------
    ValueError
        If method is not registered
    """
    impl_class = METHOD_IMPLEMENTATIONS.get(method)

    if impl_class is None:
        registered = ", ".join(m.value for m in METHOD_IMPLEMENTATIONS)
        raise ValueError(
            f"Unsupported comparison method: {method.value}. "
            f"Registered methods: {registered}"
        )

    return impl_class()
