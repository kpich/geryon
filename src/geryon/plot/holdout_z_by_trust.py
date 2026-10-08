"""Plot replication z by the critic's trustworthiness rating.

y is ``replication_z``: |explore − validation| over the SE of the difference (log
scale for ratios). It scores a null and an effect alike. A careful null that the
rerun confirms sits near 0, where validation significance would have put it at the
bottom and called it a failure. Above the dashed 1.96 line the rerun disagrees with
the explore estimate. Filled points were significant on explore (p<0.05), hollow ones
were nulls. Box plot plus jittered points per rating. Reruns with no z (no effect or
SE on either split, or a degenerate ratio) are counted above each group.
"""

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._critiques import COLORS, YTICK_LABELS
from geryon.plot._holdout import parse_args
from geryon.plot._save import save

_ALPHA = 0.05
_Z_LINE = 1.96


def main() -> None:
    args = parse_args("Plot replication z by critic trustworthiness")
    table = pd.read_csv(args.table)
    table = table[table["trustworthiness"].notna()]

    ratings = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(5, 4))
    rng = np.random.default_rng(42)
    for r in ratings:
        group = table[table["trustworthiness"] == r]
        scored = group[group["replication_z"].notna()]
        zs = scored["replication_z"].to_numpy(dtype=float)
        if len(zs):
            ax.boxplot(
                [zs],
                positions=[r],
                widths=0.5,
                showfliers=False,
                medianprops={"color": "black"},
            )
        effect = (scored["explore_p"] < _ALPHA).to_numpy()
        xs = r + rng.uniform(-0.12, 0.12, size=len(zs))
        color = COLORS["trustworthiness"][r]
        ax.scatter(xs[effect], zs[effect], color=color, alpha=0.7, s=22, zorder=3)
        ax.scatter(
            xs[~effect],
            zs[~effect],
            facecolors="none",
            edgecolors=color,
            alpha=0.9,
            s=22,
            zorder=3,
        )
        missed = int((zs > _Z_LINE).sum())
        no_z = len(group) - len(zs)
        ax.annotate(
            f"N={len(zs)}, {missed} >{_Z_LINE}" + (f"\n+{no_z} no z" if no_z else ""),
            (r, 1),
            xytext=(0, 6),
            textcoords="offset points",
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(_Z_LINE, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(ratings, [YTICK_LABELS["trustworthiness"][r] for r in ratings])
    ax.set_xlim(0.4, 3.6)
    z_max = table["replication_z"].max()
    ax.set_ylim(0, max(_Z_LINE if pd.isna(z_max) else z_max, _Z_LINE) * 1.25)
    ax.set_xlabel("Critic trustworthiness")
    ax.set_ylabel("Replication z  |explore − validation| / SE")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color="0.3", label="explore p<0.05"
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="0.3",
                label="explore null",
            ),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
    )

    plt.tight_layout()
    save(args.output)


if __name__ == "__main__":
    main()
