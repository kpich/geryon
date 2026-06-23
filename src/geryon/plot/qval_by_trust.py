"""Plot val q-value (−log10) grouped by trustworthiness rating.

Each trust group (1 spurious → 3 credible) gets a see-through quantile box plus a
jittered stripplot of its hypotheses' −log10(val q-value). Hypotheses whose val
result is missing/failed (no q-value) are rugged below the axis per group, so a
group with great-looking points but lots of NaNs reads honestly.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from geryon.plot._loader import load_hypotheses

# Okabe-Ito; matches ratings_by_depth trust mapping (spurious=bad, credible=good)
_BAD = "#D55E00"  # vermillion
_NEUTRAL = "#E69F00"  # amber
_GOOD = "#0072B2"  # blue
TRUST_COLORS = {1: _BAD, 2: _NEUTRAL, 3: _GOOD}
TRUST_LABELS = {1: "1\n(spurious)", 2: "2", 3: "3\n(credible)"}
TRUST_LEVELS = [1, 2, 3]

# q-values are floored before −log10 so a perfect q=0 doesn't blow up to inf.
_Q_FLOOR = 1e-300


def prepare_data(val_df: pd.DataFrame, hyps: list) -> pd.DataFrame:
    """Join val results with trust ratings and compute −log10(val q-value).

    BH correction is applied across every hypothesis with a (successful) val
    p-value, then −log10 of that q. Rows whose val result is missing or failed get
    val_q/neg_log10_q = NaN so the plot can rug them separately.

    Returns a DataFrame with columns: hypothesis_id, trust, val_q, neg_log10_q.
    """
    trust_by_id = {h.hypothesis_id: h.effective_rating.trustworthiness for h in hyps}

    df = val_df.copy()
    df["trust"] = df["hypothesis_id"].map(trust_by_id)

    success = df.get("val_success")
    ok = df["val_p_value"].notna()
    if success is not None:
        ok &= success.astype(str) == "True"

    df["val_q"] = np.nan
    if ok.any():
        _, q_vals, _, _ = multipletests(
            df.loc[ok, "val_p_value"].astype(float).to_numpy(), method="fdr_bh"
        )
        df.loc[ok, "val_q"] = q_vals

    df["neg_log10_q"] = -np.log10(df["val_q"].clip(lower=_Q_FLOOR))
    return df[["hypothesis_id", "trust", "val_q", "neg_log10_q"]]


def plot_qval_by_trust(df: pd.DataFrame, output_path: str | Path) -> None:
    """Stripplot + quantile box of −log10(val q-value) per trust group.

    df is the output of `prepare_data`. Boxes show quantiles only (no fliers, no
    fill); points are jittered and semi-transparent; NaN-q hypotheses are rugged
    below the axis with a per-group count.
    """
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8, 6))

    finite = df["neg_log10_q"].to_numpy()
    finite = finite[np.isfinite(finite)]
    y_top = float(np.nanmax(finite)) if finite.size else 1.0
    rug_y = -0.06 * max(y_top, 1.0)

    n_total = 0
    n_nan_total = 0
    for i, level in enumerate(TRUST_LEVELS):
        color = TRUST_COLORS[level]
        grp = df[df["trust"] == level]
        vals = grp["neg_log10_q"].to_numpy()
        vals = vals[np.isfinite(vals)]
        nan_count = int(grp["neg_log10_q"].isna().sum())
        n_total += len(vals)
        n_nan_total += nan_count

        if len(vals):
            bp = ax.boxplot(
                vals,
                positions=[i],
                widths=0.5,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": color, "linewidth": 2},
                whiskerprops={"color": color, "linewidth": 1},
                capprops={"color": color, "linewidth": 1},
            )
            for box in bp["boxes"]:
                box.set_facecolor("none")
                box.set_edgecolor(color)
                box.set_linewidth(1.3)

            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                color=color,
                alpha=0.55,
                s=28,
                edgecolors="none",
                zorder=3,
            )

        if nan_count:
            jitter = rng.uniform(-0.16, 0.16, size=nan_count)
            ax.scatter(
                np.full(nan_count, i) + jitter,
                np.full(nan_count, rug_y),
                marker="|",
                s=120,
                linewidths=1.2,
                color=color,
                clip_on=False,
                zorder=4,
            )
            ax.text(
                i,
                rug_y * 1.9,
                f"NaN: {nan_count}",
                ha="center",
                va="top",
                fontsize=8,
                color=color,
            )

    ax.axhline(rug_y, color="gray", linewidth=0.6, alpha=0.4)
    ax.axhline(
        -np.log10(0.05),
        color="tab:red",
        linestyle=":",
        linewidth=1,
        label="q = 0.05",
    )

    ax.set_xticks(range(len(TRUST_LEVELS)))
    ax.set_xticklabels([TRUST_LABELS[level] for level in TRUST_LEVELS])
    ax.set_xlabel("Trustworthiness")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{val\ q\text{-}value})$")
    ax.set_title(
        f"Validation q-value by trustworthiness "
        f"(n={n_total}; {n_nan_total} with no val q)"
    )
    ax.set_ylim(bottom=rug_y * 2.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="CSV from geryon.eval.batch"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Geryon data dir (contains sessions/)",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    val_df = pd.read_csv(args.input)
    hyps = load_hypotheses(args.data_dir)
    df = prepare_data(val_df, hyps)
    plot_qval_by_trust(df, args.output)


if __name__ == "__main__":
    main()
