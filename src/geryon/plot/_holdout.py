"""Shared loading for the plots of the validation rerun (see holdout_table)."""

import argparse
from pathlib import Path

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

    Effect sizes and bounds come back on the log scale. Only rows whose rerun
    reported an effect are kept.
    """
    observed = table.loc[
        table["ratio"] & table["val_effect"].notna(), ["hypothesis_id", "val_effect"]
    ]
    df = forecasts.merge(observed, on="hypothesis_id")
    for col in ["predicted", "lower", "upper", "val_effect"]:
        df[col] = np.log(df[col].astype(float))
    return df
