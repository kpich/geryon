"""Base classes and protocols for comparison methods."""

from typing import Protocol

import pandas as pd  # type: ignore


class ComparisonMethodImpl(Protocol):
    """Protocol for statistical comparison method implementations.

    Each comparison method (Cox regression, log-rank test, etc.) has a
    corresponding implementation that knows how to calculate statistics
    from cohort data.
    """

    def calculate(
        self, cohort_a_data: pd.DataFrame, cohort_b_data: pd.DataFrame
    ) -> dict[str, float]:
        """Calculate statistical comparison between two cohorts.

        Parameters
        ----------
        cohort_a_data : pd.DataFrame
            Outcome data for cohort A (columns: PATIENT_ID, time, event)
        cohort_b_data : pd.DataFrame
            Outcome data for cohort B (columns: PATIENT_ID, time, event)

        Returns
        -------
        dict[str, float]
            Statistical results with keys like:
            - hazard_ratio, confidence_interval_lower/upper, p_value (Cox)
            - Or other method-specific statistics
        """
        ...
