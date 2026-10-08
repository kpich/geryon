"""Plot validation q-value against explore q-value, one point per hypothesis.

Both axes are −log10 of the BH q-value, with the 0.05 threshold dashed on each.
Treating the validation rerun as truth, an explore-significant hypothesis claims an
effect and an explore null claims none, and either can fail: an effect that is not
significant on validation (bottom right), or a null that is (top left). Points are
drawn as the critic's trustworthiness rating (1-3, "–" if unrated), black where the
rerun agreed and red where it didn't. Counts per quadrant sit in the corners, and
the failure rate of each kind of claim per rating beside the plot. Reruns that
reported no p-value can't be scored and are only counted.

Significance is a coarse test of a null, since an underpowered rerun "confirms" one by
default; ``holdout_z_by_trust`` scores agreement on the effect itself.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geryon.plot._critiques import BAD
from geryon.plot._holdout import parse_args
from geryon.plot._save import save

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
    agreed = df["explore_sig"] == df["val_sig"]
    for xi, yi, ok, trust in zip(x, y, agreed, df["trustworthiness"], strict=True):
        label = "–" if pd.isna(trust) else str(int(trust))
        ax.scatter(
            xi,
            yi,
            marker=f"${label}$",
            s=70,
            color="black" if ok else BAD,
            zorder=3,
        )
    ax.axvline(line, color="black", linestyle="--", linewidth=1)
    ax.axhline(line, color="black", linestyle="--", linewidth=1)

    quadrants = {
        "effect replicated": (
            df["explore_sig"] & df["val_sig"],
            (0.98, 0.98),
            "right",
            "top",
        ),
        "effect failed": (
            df["explore_sig"] & ~df["val_sig"],
            (0.98, 0.02),
            "right",
            "bottom",
        ),
        "null failed": (
            ~df["explore_sig"] & df["val_sig"],
            (0.02, 0.98),
            "left",
            "top",
        ),
        "null replicated": (
            ~df["explore_sig"] & ~df["val_sig"],
            (0.02, 0.02),
            "left",
            "bottom",
        ),
    }
    for name, (mask, xy, ha, va) in quadrants.items():
        ax.annotate(
            f"{name} {int(mask.sum())}",
            xy,
            xycoords="axes fraction",
            ha=ha,
            va=va,
            fontsize=9,
            color=BAD if name.endswith("failed") else "black",
        )

    def rates(claims: pd.DataFrame, failed: pd.Series) -> list[str]:
        return [
            f"  trust {r:.0f}: {int(failed[g.index].sum())}/{len(g)}"
            for r, g in claims.groupby("trustworthiness")
        ]

    effects = df[df["explore_sig"]]
    nulls = df[~df["explore_sig"]]
    ax.text(
        1.03,
        0.0,
        "Effects failed\n"
        + "\n".join(rates(effects, ~df["val_sig"]))
        + "\nNulls failed\n"
        + "\n".join(rates(nulls, df["val_sig"])),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )

    n_effect_failed = int(quadrants["effect failed"][0].sum())
    n_null_failed = int(quadrants["null failed"][0].sum())
    title = (
        f"Failed on validation at q<{_Q_THRESHOLD}: {n_effect_failed}/{len(effects)} "
        f"effects, {n_null_failed}/{len(nulls)} nulls"
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
    save(args.output)


if __name__ == "__main__":
    main()
