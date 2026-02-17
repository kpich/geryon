"""Human feedback labels for hypothesis quality."""

from enum import Enum


class HypothesisLabel(str, Enum):
    """Human feedback labels for hypothesis quality.

    Labels guide learning from hypothesis results and avoid repeating mistakes.
    """

    CORRECT = "correct"
    """Valid, interesting finding worth following up."""

    RED_HERRING = "red_herring"
    """Spurious correlation, not causally meaningful."""

    CONFOUNDED = "confounded"
    """Missing key confounders that invalidate comparison."""

    DATA_ISSUE = "data_issue"
    """Data quality problem (missingness, errors, bias)."""

    DUPLICATE = "duplicate"
    """Already tested or trivial variant of previous hypothesis."""

    NOT_NOVEL = "not_novel"
    """Known or obvious result, not informative."""

    WRONG_SPEC = "wrong_spec"
    """Spec filters don't capture the intended cohort semantics."""

    PENDING = "pending"
    """Not yet labeled by human reviewer."""
