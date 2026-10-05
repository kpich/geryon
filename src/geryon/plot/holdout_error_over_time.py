"""Plot forecast error on the validation rerun over the run, critic vs explore baseline.

x is each hypothesis's position in creation order among all rerun hypotheses, so
gaps mark ones with no validation effect. Faint points are per-hypothesis
|forecast − validation| in log units; the lines are rolling means. If the critic
learns to spot overfitting as chains refine, its line should fall below the
baseline's. Ratio effects only.
"""

import matplotlib.pyplot as plt
import pandas as pd

from geryon.plot._holdout import (
    METHOD_COLORS,
    METHOD_LABELS,
    parse_args,
    ratio_forecasts,
)

_METHODS = ["critic", "explore"]


def main() -> None:
    args = parse_args("Plot validation forecast error over time")
    table = pd.read_csv(args.table).sort_values("created_at").reset_index(drop=True)
    order = pd.Series(table.index, index=table["hypothesis_id"], name="seq")
    df = ratio_forecasts(table, pd.read_csv(args.forecasts)).join(
        order, on="hypothesis_id"
    )
    df["abs_error"] = (df["predicted"] - df["val_effect"]).abs()
    paired = df.pivot(index="seq", columns="method", values="abs_error")[_METHODS]
    paired = paired.dropna().sort_index()

    window = max(3, len(paired) // 5)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for method in _METHODS:
        color = METHOD_COLORS[method]
        ax.scatter(paired.index, paired[method], color=color, alpha=0.3, s=16)
        rolling = paired[method].rolling(window).mean()
        ax.plot(
            paired.index,
            rolling,
            color=color,
            linewidth=2,
            label=f"{METHOD_LABELS[method]}  (mean {paired[method].mean():.2f})",
        )
    ax.set_xlabel("Hypothesis, in creation order")
    ax.set_ylabel("|forecast − validation|  (log units)")
    ax.set_title(f"Rolling mean, window {window}, N={len(paired)}", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
