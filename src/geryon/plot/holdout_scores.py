"""Is the critic's forecast of the held-out effect better than the naive baselines?

Scored over the hypotheses where both the critic and the explore estimate gave an
80% interval, so the comparison is paired. Left: mean absolute error in log units,
for the critic, the explore estimate and no effect. Right: mean 80% interval score
(width, plus 2/α times any miss), which rewards narrow intervals and punishes ones
that miss; no effect has no interval. Bars carry 95% bootstrap intervals, and each
panel's title gives the paired difference critic − explore estimate with its own
bootstrap interval: negative means the critic was better. Ratio effects only.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._holdout import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    parse_args,
    ratio_forecasts,
)
from geryon.plot.holdout_table import LEVEL

_BOOT = 4000


def interval_score(
    lower: np.ndarray, upper: np.ndarray, y: np.ndarray, level: float = LEVEL
) -> np.ndarray:
    """Gneiting-Raftery score of a central `level` interval; lower is better."""
    alpha = 1 - level
    below = np.clip(lower - y, 0, None)
    above = np.clip(y - upper, 0, None)
    return (upper - lower) + (2 / alpha) * (below + above)


def bootstrap_mean(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(_BOOT, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    args = parse_args("Score the critic's forecast against naive baselines")
    df = ratio_forecasts(pd.read_csv(args.table), pd.read_csv(args.forecasts))
    has_interval = df.loc[df["lower"].notna()].groupby("hypothesis_id")["method"]
    paired = has_interval.nunique()
    df = df[df["hypothesis_id"].isin(paired[paired == 2].index)]
    wide = df.pivot(index="hypothesis_id", columns="method")
    y = wide[("val_effect", "critic")].to_numpy()

    errors = {m: np.abs(wide[("predicted", m)].to_numpy() - y) for m in METHODS}
    scores = {
        m: interval_score(
            wide[("lower", m)].to_numpy(), wide[("upper", m)].to_numpy(), y
        )
        for m in ["critic", "explore"]
    }

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [
        (axes[0], errors, "Mean absolute error  (log units)"),
        (axes[1], scores, f"Mean {LEVEL:.0%} interval score  (log units)"),
    ]
    for ax, values, ylabel in panels:
        methods = list(values)
        xs = np.arange(len(methods))
        for x, m in zip(xs, methods, strict=True):
            mean = values[m].mean()
            lo, hi = bootstrap_mean(values[m], rng)
            ax.bar(x, mean, color=METHOD_COLORS[m], alpha=0.7, width=0.6)
            ax.errorbar(
                x, mean, yerr=[[mean - lo], [hi - mean]], color="black", capsize=4
            )
        diff = values["critic"] - values["explore"]
        lo, hi = bootstrap_mean(diff, rng)
        ax.set_title(
            f"critic − explore: {diff.mean():+.2f}  [{lo:+.2f}, {hi:+.2f}]",
            fontsize=10,
        )
        ax.set_xticks(xs, [METHOD_LABELS[m] for m in methods], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        f"Paired over N={len(y)} hypotheses with a critic forecast", fontsize=10
    )

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
