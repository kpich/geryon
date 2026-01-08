"""Comparison method definitions for hypothesis testing."""

from enum import Enum


class ComparisonMethod(str, Enum):
    """Statistical method for comparing cohorts."""

    HAZARD_RATIO_COX = "hazard_ratio_cox"
