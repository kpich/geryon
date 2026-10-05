"""Plot the critic's forecast against the explore-estimate baseline, paired.

Three panels, each point one hypothesis, each with the identity line:

- Point forecast. A point between the diagonal and y=0 is a forecast the critic
  shrank toward no effect.
- 80% interval width.
- Absolute error against the validation effect. Below the diagonal, the critic was
  closer.

Ratio effects only, on the log scale.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._holdout import METHOD_COLORS, parse_args, ratio_forecasts


def _identity(ax, xs: np.ndarray, ys: np.ndarray) -> None:
    both = np.concatenate([xs, ys])
    pad = 0.05 * (both.max() - both.min())
    lim = (both.min() - pad, both.max() + pad)
    ax.plot(lim, lim, color="black", linewidth=1, alpha=0.5, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = parse_args("Plot critic forecast vs explore baseline, paired")
    df = ratio_forecasts(pd.read_csv(args.table), pd.read_csv(args.forecasts))
    wide = df.pivot(index="hypothesis_id", columns="method")
    wide = wide[wide[("predicted", "critic")].notna()]

    def col(field: str, method: str) -> np.ndarray:
        return wide[(field, method)].to_numpy(dtype=float)

    observed = col("val_effect", "critic")
    panels = [
        (
            "Point forecast  log(effect)",
            col("predicted", "explore"),
            col("predicted", "critic"),
        ),
        (
            "80% interval width  (log units)",
            col("upper", "explore") - col("lower", "explore"),
            col("upper", "critic") - col("lower", "critic"),
        ),
        (
            "|forecast − validation|  (log units)",
            np.abs(col("predicted", "explore") - observed),
            np.abs(col("predicted", "critic") - observed),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (label, xs, ys) in zip(axes, panels, strict=True):
        ax.scatter(xs, ys, color=METHOD_COLORS["critic"], s=22, alpha=0.8, zorder=3)
        _identity(ax, xs, ys)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Baseline: explore estimate")
        ax.set_ylabel("Critic")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5)
    closer = int((panels[2][2] < panels[2][1]).sum())
    axes[2].set_title(
        f"{panels[2][0]}\ncritic closer on {closer}/{len(observed)}", fontsize=10
    )

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
