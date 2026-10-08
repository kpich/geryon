"""Plot forecast error on the validation rerun over the run, critic vs explore baseline.

x is each hypothesis's position in creation order among all rerun hypotheses, so
gaps mark ones with no validation effect. Faint points are per-hypothesis
fold errors, max(forecast/validation, validation/forecast); the lines are rolling
geometric means. If the critic
learns to spot overfitting as chains refine, its line should fall below the
baseline's. Ratio effects only.
"""

import matplotlib.pyplot as plt
import pandas as pd

from geryon.plot._holdout import (
    METHOD_COLORS,
    METHOD_LABELS,
    fold,
    fold_axis,
    parse_args,
    ratio_forecasts,
)
from geryon.plot._save import save

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
            label=f"{METHOD_LABELS[method]}  (typical {fold(paired[method].mean())})",
        )
    ax.set_xlabel("Hypothesis, in creation order")
    ax.set_ylabel("Fold error vs validation")
    fold_axis(ax.yaxis)
    ax.set_title(
        f"Rolling geometric mean, window {window}, N={len(paired)}", fontsize=10
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save(args.output)


if __name__ == "__main__":
    main()
