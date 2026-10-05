"""Forest plot: explore estimate, critic forecast and validation rerun per hypothesis.

One row per ratio hypothesis, sorted by explore |z|. Each row has three intervals:
the explore 95% CI (the generator's own result), the critic's 80% forecast of the
held-out effect, and the validation 95% CI. A critic that sees overfitting should
pull its interval toward 1 on the rows where validation lands there too. Reruns that
reported no effect are marked at the right. The x-axis is clipped, so a few very wide
validation CIs run off the edge.
"""

import math

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._holdout import METHOD_COLORS, parse_args
from geryon.plot.holdout_table import explore_se

_TITLE_CHARS = 70
_OFFSET = 0.25
_XLIM = (0.05, 20)


def _short(title: str) -> str:
    return title if len(title) <= _TITLE_CHARS else title[: _TITLE_CHARS - 1] + "…"


def main() -> None:
    args = parse_args("Forest plot of explore, critic forecast and validation effects")
    table = pd.read_csv(args.table)
    forecasts = pd.read_csv(args.forecasts)
    df = table[table["ratio"] & table["explore_effect"].notna()].copy()
    critic = forecasts[forecasts["method"] == "critic"].set_index("hypothesis_id")

    def z(r: pd.Series) -> float:
        se = explore_se(
            r["explore_effect"],
            None if pd.isna(r["explore_lower"]) else r["explore_lower"],
            None if pd.isna(r["explore_upper"]) else r["explore_upper"],
            None if pd.isna(r["explore_p"]) else r["explore_p"],
            ratio=True,
        )
        return abs(math.log(r["explore_effect"])) / se if se else 0.0

    df["z"] = df.apply(z, axis=1)
    df = df.sort_values("z").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(df) + 1.2))
    explore_color, critic_color, val_color = "0.45", METHOD_COLORS["critic"], "black"
    for y, (_, r) in enumerate(df.iterrows()):
        if pd.notna(r["explore_lower"]):
            ax.plot(
                [r["explore_lower"], r["explore_upper"]],
                [y + _OFFSET] * 2,
                color=explore_color,
                linewidth=2,
            )
        ax.scatter(r["explore_effect"], y + _OFFSET, color=explore_color, s=14)

        hid = r["hypothesis_id"]
        if hid in critic.index:
            c = critic.loc[hid]
            if pd.notna(c["lower"]):
                ax.plot(
                    [c["lower"], c["upper"]], [y] * 2, color=critic_color, linewidth=2
                )
            ax.scatter(c["predicted"], y, color=critic_color, s=14)

        if pd.notna(r["val_effect"]):
            if pd.notna(r["val_lower"]) and pd.notna(r["val_upper"]):
                ax.plot(
                    [r["val_lower"], r["val_upper"]],
                    [y - _OFFSET] * 2,
                    color=val_color,
                    linewidth=2,
                )
            ax.scatter(r["val_effect"], y - _OFFSET, color=val_color, s=14)
        else:
            ax.annotate(
                "no validation effect",
                (1.0, y - _OFFSET),
                xycoords=("axes fraction", "data"),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=7,
                color="0.4",
            )

    ax.axvline(1, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlim(*_XLIM)
    ax.set_yticks(
        np.arange(len(df)),
        [
            f"{_short(t)}  (z={zv:.1f})"
            for t, zv in zip(df["title"], df["z"], strict=True)
        ],
        fontsize=7,
    )
    ax.set_ylim(-0.7, len(df) - 0.3)
    ax.set_xlabel("Effect (ratio, log scale)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(
        handles=[
            Line2D([], [], color=explore_color, lw=2, label="Explore 95% CI"),
            Line2D([], [], color=critic_color, lw=2, label="Critic 80% forecast"),
            Line2D([], [], color=val_color, lw=2, label="Validation 95% CI"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
