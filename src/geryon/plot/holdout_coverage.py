"""Plot how well the 80% forecast intervals cover the validation effect.

Left: share of validation effects inside each method's interval, with a Wilson 95%
interval; the dashed line is the nominal 0.8. Right: interval widths in log units,
since a wide enough interval always covers. Ratio effects only.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import (  # type: ignore[import-untyped]
    proportion_confint,
)

from geryon.plot._holdout import (
    METHOD_COLORS,
    METHOD_LABELS,
    parse_args,
    ratio_forecasts,
)
from geryon.plot.holdout_table import LEVEL

_INTERVAL_METHODS = ["critic", "explore"]


def main() -> None:
    args = parse_args("Plot forecast interval coverage on the validation rerun")
    df = ratio_forecasts(pd.read_csv(args.table), pd.read_csv(args.forecasts))
    df = df[df["lower"].notna()]

    fig, (ax_cov, ax_w) = plt.subplots(1, 2, figsize=(9, 3.6))
    xs = np.arange(len(_INTERVAL_METHODS))
    widths = []
    for x, method in zip(xs, _INTERVAL_METHODS, strict=True):
        m = df[df["method"] == method]
        n = len(m)
        hits = int(
            ((m["lower"] <= m["val_effect"]) & (m["val_effect"] <= m["upper"])).sum()
        )
        color = METHOD_COLORS[method]
        if n:
            lo, hi = proportion_confint(hits, n, method="wilson")
            ax_cov.bar(x, hits / n, color=color, alpha=0.7, width=0.6)
            ax_cov.errorbar(
                x,
                hits / n,
                yerr=[[hits / n - lo], [hi - hits / n]],
                color="black",
                capsize=4,
            )
        ax_cov.text(x, 1.02, f"{hits}/{n}", ha="center", va="bottom", fontsize=9)
        widths.append((m["upper"] - m["lower"]).to_numpy())

    ax_cov.axhline(LEVEL, color="black", linestyle="--", linewidth=1)
    ax_cov.set_xticks(xs, [METHOD_LABELS[m] for m in _INTERVAL_METHODS], fontsize=9)
    ax_cov.set_ylim(0, 1.12)
    ax_cov.set_ylabel(f"Coverage of {LEVEL:.0%} interval")
    ax_cov.grid(True, alpha=0.3, axis="y")

    ax_w.boxplot(
        widths,
        positions=xs,
        widths=0.5,
        showfliers=False,
        medianprops={"color": "black"},
    )
    rng = np.random.default_rng(42)
    for x, method, w in zip(xs, _INTERVAL_METHODS, widths, strict=True):
        ax_w.scatter(
            x + rng.uniform(-0.12, 0.12, size=len(w)),
            w,
            color=METHOD_COLORS[method],
            alpha=0.6,
            s=18,
            zorder=3,
        )
    ax_w.set_xticks(xs, [METHOD_LABELS[m] for m in _INTERVAL_METHODS], fontsize=9)
    ax_w.set_ylabel("Interval width  (log units)")
    ax_w.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
