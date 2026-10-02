"""Plot forecast vs observed validation effect, one panel per forecasting method.

Ratio effects only (HRs, ORs), on the log scale. Horizontal bars are each method's
80% interval. The panel title gives the mean absolute error in log units.
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


def main() -> None:
    args = parse_args("Plot forecast vs observed validation effect")
    df = ratio_forecasts(pd.read_csv(args.table), pd.read_csv(args.forecasts))

    vals = df[["predicted", "lower", "upper", "val_effect"]].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    pad = 0.1
    lim = (finite.min() - pad, finite.max() + pad) if finite.size else (-1, 1)

    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(4 * len(METHODS), 4.2), sharex=True, sharey=True
    )
    for ax, method in zip(axes, METHODS, strict=True):
        m = df[df["method"] == method]
        color = METHOD_COLORS[method]
        has_iv = m["lower"].notna()
        ax.errorbar(
            m.loc[has_iv, "predicted"],
            m.loc[has_iv, "val_effect"],
            xerr=[
                m.loc[has_iv, "predicted"] - m.loc[has_iv, "lower"],
                m.loc[has_iv, "upper"] - m.loc[has_iv, "predicted"],
            ],
            fmt="none",
            ecolor=color,
            alpha=0.35,
            zorder=2,
        )
        # Every no-effect forecast sits at 0; spread them so they don't stack.
        spread = 0.03 if method == "no_effect" else 0.0
        jitter = np.random.default_rng(42).uniform(-spread, spread, size=len(m))
        ax.scatter(
            m["predicted"] + jitter, m["val_effect"], color=color, s=22, zorder=3
        )
        ax.plot(lim, lim, color="black", linewidth=1, alpha=0.5, zorder=1)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        mae = (m["predicted"] - m["val_effect"]).abs().mean()
        ax.set_title(f"{METHOD_LABELS[method]}\nN={len(m)}  MAE={mae:.2f}", fontsize=10)
        ax.set_xlabel("Forecast  log(effect)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation rerun  log(effect)")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
