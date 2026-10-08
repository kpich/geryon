"""Shared loading for the plots of the validation rerun (see holdout_table)."""

import argparse
from pathlib import Path

from matplotlib.axis import Axis
from matplotlib.ticker import AutoLocator, FuncFormatter, Locator
import numpy as np
import pandas as pd

from geryon.plot._critiques import BAD, GOOD, NEUTRAL

METHODS = ["critic", "expectation", "explore", "no_effect"]
METHOD_LABELS = {
    "critic": "Critic forecast",
    "explore": "Baseline: explore estimate",
    "no_effect": "Baseline: no effect",
    "expectation": "Blind expectation",
}
METHOD_COLORS = {
    "critic": GOOD,
    "explore": NEUTRAL,
    "no_effect": BAD,
    "expectation": "#009E73",
}


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--forecasts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def ratio_forecasts(table: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Forecasts of ratio effects joined to the observed validation effect.

    Effect sizes and bounds come back on the log scale, where a ratio's errors are
    symmetric; label the axes with ``ratio_axis`` and ``fold_axis`` so readers see
    ratios rather than logs. Only rows whose rerun reported an effect are kept.
    """
    observed = table.loc[
        table["ratio"] & table["val_effect"].notna(), ["hypothesis_id", "val_effect"]
    ]
    df = forecasts.merge(observed, on="hypothesis_id")
    for col in ["predicted", "lower", "upper", "val_effect"]:
        df[col] = np.log(df[col].astype(float))
    return df


# Tick candidates, finest first. A locator takes the finest set that puts a readable
# number of ticks in view and spreads them across most of it; the fine sets stop at
# small values, so on a wide axis they would bunch at one end.
_RATIO_TICKS: list[list[float]] = [
    [0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2],
    [0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4],
    [0.1, 0.2, 0.5, 1, 2, 5, 10],
    [0.01, 0.1, 1, 10, 100],
]
_FOLD_TICKS: list[list[float]] = [
    [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2],
    [1, 1.25, 1.5, 2, 2.5, 3],
    [1, 1.5, 2, 3, 5, 10],
    [1, 2, 5, 10, 100],
]


class _NiceLogLocator(Locator):
    """Ticks at log(v) for round values v, on an axis that holds logs."""

    def __init__(self, tiers: list[list[float]]) -> None:
        self.tiers = tiers

    def __call__(self) -> list[float]:
        assert self.axis is not None
        lo, hi = sorted(self.axis.get_view_interval())
        fallback: list[float] = []
        for tier in self.tiers:
            ticks = [float(np.log(v)) for v in tier if lo <= np.log(v) <= hi]
            spread = ticks[-1] - ticks[0] if ticks else 0.0
            if 3 <= len(ticks) <= 8 and spread >= 0.6 * (hi - lo):
                return ticks
            if len(ticks) >= 2:
                fallback = ticks
        if fallback:
            return fallback
        auto = AutoLocator()
        auto.set_axis(self.axis)
        return list(auto())


def ratio_axis(axis: Axis) -> None:
    """Label an axis holding log(ratio) with the ratio itself (0.5, 1, 2)."""
    axis.set_major_locator(_NiceLogLocator(_RATIO_TICKS))
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{np.exp(x):.3g}"))


def fold_axis(axis: Axis) -> None:
    """Label an axis holding a log-scale difference as a fold change (1.5×, 2×).

    |log a − log b| is log(max(a/b, b/a)), so an error of 0.4 reads as 1.5×.
    """
    axis.set_major_locator(_NiceLogLocator(_FOLD_TICKS))
    axis.set_major_formatter(FuncFormatter(lambda x, _: f"{np.exp(x):.3g}×"))


def fold(log_diff: float) -> str:
    """A mean log-scale difference as its geometric-mean fold change."""
    return f"{np.exp(log_diff):.2f}×"
