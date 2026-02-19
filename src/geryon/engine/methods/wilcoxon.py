"""Wilcoxon rank-sum (Mann-Whitney U) method implementation."""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.stats import mannwhitneyu  # type: ignore


class WilcoxonRankSumMethod:
    """Wilcoxon rank-sum test for comparing continuous outcomes."""

    def calculate(
        self, cohort_a_data: pd.DataFrame, cohort_b_data: pd.DataFrame
    ) -> dict[str, float | None]:
        """Calculate Mann-Whitney U statistic and medians.

        Parameters
        ----------
        cohort_a_data : pd.DataFrame
            Must have column: value
        cohort_b_data : pd.DataFrame
            Must have column: value

        Returns
        -------
        dict with keys: median_a, median_b, u_statistic, p_value
        """
        a = cohort_a_data["value"].dropna()
        b = cohort_b_data["value"].dropna()

        if len(a) == 0 or len(b) == 0:
            return {
                "median_a": None,
                "median_b": None,
                "u_statistic": None,
                "p_value": None,
            }

        u_stat, p_value = mannwhitneyu(a, b, alternative="two-sided")

        return {
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
        }
