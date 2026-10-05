"""Plot validation q-value by the critic's trustworthiness rating.

Box plot plus jittered points of −log10 of the BH q-value from the validation rerun, per
rating. Reruns that reported no p-value (script failed, or no test) are counted
above each group rather than plotted.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._critiques import COLORS, YTICK_LABELS
from geryon.plot._holdout import parse_args

_Q_THRESHOLD = 0.05


def main() -> None:
    args = parse_args("Plot validation q-value by critic trustworthiness")
    table = pd.read_csv(args.table)
    table = table[table["trustworthiness"].notna()]

    ratings = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(5, 4))
    rng = np.random.default_rng(42)
    for r in ratings:
        group = table[table["trustworthiness"] == r]
        qs = -np.log10(group["val_q"].dropna().to_numpy(dtype=float))
        if len(qs):
            ax.boxplot(
                [qs],
                positions=[r],
                widths=0.5,
                showfliers=False,
                medianprops={"color": "black"},
            )
        ax.scatter(
            r + rng.uniform(-0.12, 0.12, size=len(qs)),
            qs,
            color=COLORS["trustworthiness"][r],
            alpha=0.7,
            s=22,
            zorder=3,
        )
        no_p = len(group) - len(qs)
        ax.annotate(
            f"N={len(qs)}" + (f"\n+{no_p} no p" if no_p else ""),
            (r, 1),
            xytext=(0, 6),
            textcoords="offset points",
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(-np.log10(_Q_THRESHOLD), color="black", linestyle="--", linewidth=1)
    ax.set_xticks(ratings, [YTICK_LABELS["trustworthiness"][r] for r in ratings])
    ax.set_xlim(0.4, 3.6)
    ax.set_xlabel("Critic trustworthiness")
    ax.set_ylabel("Validation rerun  −log10 q (BH)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
