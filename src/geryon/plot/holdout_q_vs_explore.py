"""Plot validation q-value against explore q-value, one point per hypothesis.

Both axes are −log10 of the BH q-value, with the 0.05 threshold dashed on each.
Treating the validation rerun as truth, a point right of the vertical line and below
the horizontal one is a false positive: significant on explore, not on validation.
Points are drawn as the critic's trustworthiness rating (1-3, "–" if unrated),
black if significant on validation and gray if not. Counts per quadrant sit in the
corners, and the false positive rate per rating beside the plot. Reruns that reported no
p-value can't be scored and are only counted.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._holdout import parse_args

_Q_THRESHOLD = 0.05


def main() -> None:
    args = parse_args("Plot validation q-value against explore q-value")
    table = pd.read_csv(args.table)
    table = table[table["explore_q"].notna()]
    no_val = table[table["val_q"].isna()]
    df = table[table["val_q"].notna()].copy()
    df["explore_sig"] = df["explore_q"] < _Q_THRESHOLD
    df["val_sig"] = df["val_q"] < _Q_THRESHOLD
    x = -np.log10(df["explore_q"].to_numpy(dtype=float))
    y = -np.log10(df["val_q"].to_numpy(dtype=float))
    line = -np.log10(_Q_THRESHOLD)

    fig, ax = plt.subplots(figsize=(6, 5))
    for xi, yi, sig, trust in zip(
        x, y, df["val_sig"], df["trustworthiness"], strict=True
    ):
        label = "–" if pd.isna(trust) else str(int(trust))
        ax.scatter(
            xi,
            yi,
            marker=f"${label}$",
            s=70,
            color="black" if sig else "0.65",
            zorder=3,
        )
    ax.axvline(line, color="black", linestyle="--", linewidth=1)
    ax.axhline(line, color="black", linestyle="--", linewidth=1)

    quadrants = {
        "TP": (df["explore_sig"] & df["val_sig"], (0.98, 0.98), "right", "top"),
        "FP": (df["explore_sig"] & ~df["val_sig"], (0.98, 0.02), "right", "bottom"),
        "FN": (~df["explore_sig"] & df["val_sig"], (0.02, 0.98), "left", "top"),
        "TN": (~df["explore_sig"] & ~df["val_sig"], (0.02, 0.02), "left", "bottom"),
    }
    for name, (mask, xy, ha, va) in quadrants.items():
        ax.annotate(
            f"{name} {int(mask.sum())}",
            xy,
            xycoords="axes fraction",
            ha=ha,
            va=va,
            fontsize=9,
            color="black" if name in ("TP", "FN") else "0.4",
        )

    called = df[df["explore_sig"]]
    by_trust = [
        f"trust {r:.0f}: {int((~g['val_sig']).sum())}/{len(g)}"
        for r, g in called.groupby("trustworthiness")
    ]
    ax.text(
        1.03,
        0.0,
        "FP / explore-significant\n" + "\n".join(by_trust),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    n_fp = int(quadrants["FP"][0].sum())
    n_called = len(called)
    title = (
        f"False positives (validation as truth): {n_fp}/{n_called} "
        f"explore-significant at q<{_Q_THRESHOLD}"
    )
    if len(no_val):
        untested = int((no_val["explore_q"] < _Q_THRESHOLD).sum())
        title += (
            f"\n+{len(no_val)} reruns with no validation p "
            f"({untested} explore-significant), not plotted"
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Explore  −log10 q (BH)")
    ax.set_ylabel("Validation rerun  −log10 q (BH)")
    # Explore q runs to ~1e-20, which would crowd everything else against the line.
    ax.set_xscale("symlog", linthresh=2)
    ticks = [t for t in [0, 1, 2, 5, 10, 20, 50, 100] if t <= x.max() * 1.2]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(0, x.max() * 1.2)
    ax.set_ylim(0, max(y.max(), line) * 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
