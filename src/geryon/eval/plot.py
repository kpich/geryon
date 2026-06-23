"""Plotting utilities for hypothesis evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

_N_BOOT = 500
_LOESS_FRAC = 0.75

# Need at least this many paired points for OLS/LOESS/KDE and BH correction.
_MIN_POINTS = 2

# Hypotheses with a train result but no val result (val execution failed) get a
# known x but no y, so they're rugged along the bottom instead of being dropped.
_MISSING_COLOR = "#D55E00"  # Okabe-Ito vermillion


def _save_placeholder(output_path: str | Path, message: str) -> None:
    """Write a small figure with an explanatory message (too few points to plot)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _loess_with_bootstrap_ci(
    x: np.ndarray, y: np.ndarray, x_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_loess, ci_lo, ci_hi) on x_grid."""
    loess_orig = lowess(y, x, frac=_LOESS_FRAC, return_sorted=True)
    y_loess = np.interp(x_grid, loess_orig[:, 0], loess_orig[:, 1])
    rng = np.random.default_rng(42)
    boot_curves = np.empty((_N_BOOT, len(x_grid)))
    for i in range(_N_BOOT):
        idx = rng.integers(0, len(x), size=len(x))
        fit = lowess(y[idx], x[idx], frac=_LOESS_FRAC, return_sorted=True)
        boot_curves[i] = np.interp(x_grid, fit[:, 0], fit[:, 1])
    return (
        y_loess,
        np.percentile(boot_curves, 2.5, axis=0),
        np.percentile(boot_curves, 97.5, axis=0),
    )


def _marginal_bin_edges(n: int) -> np.ndarray:
    """Bin edges over [0, 1] for a marginal histogram of `n` points.

    Bin count grows with sample size (~2·√n) so the marginals get finer as more
    data accumulates, clamped to a sane [20, 80] range. p/q-values live in [0, 1],
    so the edges are fixed to that span rather than the data's min/max.
    """
    n_bins = int(np.clip(round(2 * np.sqrt(max(n, 1))), 20, 80))
    return np.linspace(0.0, 1.0, n_bins + 1)


def _make_marginal_fig(
    x: np.ndarray, y: np.ndarray
) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    """Create a figure with marginal histogram axes.

    Returns (fig, ax_main, ax_top, ax_right).
    ax_top shows the distribution of x; ax_right shows the distribution of y.
    """
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[5, 1], height_ratios=[1, 5], hspace=0.05, wspace=0.05
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    color = "steelblue"
    for vals, axis, horizontal in [(x, ax_top, True), (y, ax_right, False)]:
        if len(vals) == 0:
            continue
        edges = _marginal_bin_edges(len(vals))
        axis.hist(
            vals,
            bins=edges,
            density=True,
            color=color,
            alpha=0.4,
            edgecolor=color,
            linewidth=0.4,
            orientation="vertical" if horizontal else "horizontal",
        )

    ax_top.tick_params(labelbottom=False, left=False, labelleft=False)
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_right.tick_params(labelleft=False, bottom=False, labelbottom=False)
    ax_right.spines[["top", "right", "bottom"]].set_visible(False)

    return fig, ax, ax_top, ax_right


def _plot_scatter_with_fits(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Populate ax with scatter, identity line, OLS fit, and LOESS+CI."""
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    r2 = r_value**2

    x_grid = np.linspace(0.0, 1.0, 200)
    y_loess, ci_lo, ci_hi = _loess_with_bootstrap_ci(x, y, x_grid)

    n_above = int(np.sum(y > x))
    n_below = int(np.sum(y < x))

    ax.scatter(x, y, alpha=0.6, edgecolors="none", s=35, zorder=3)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="identity")
    ax.plot(
        x_grid,
        slope * x_grid + intercept,
        color="tab:orange",
        linewidth=1.5,
        label=f"OLS  R²={r2:.3f}  p={p_value:.3g}",
    )
    ax.plot(x_grid, y_loess, color="tab:blue", linewidth=1.5, label="LOESS")
    ax.fill_between(
        x_grid, ci_lo, ci_hi, color="tab:blue", alpha=0.15, label="95% CI (bootstrap)"
    )

    ax.text(
        0.03,
        0.97,
        f"above identity (val worse): {n_above}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="gray",
    )
    ax.text(
        0.03,
        0.91,
        f"below identity (val better): {n_below}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="gray",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _plot_qval_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Scatter train vs val q-values with just the identity line — no OLS/LOESS fit.

    Points significant in train OR val (q ≤ 0.05) keep full color; everything else
    is grayed out so the hits stand out.
    """
    sig = (x <= 0.05) | (y <= 0.05)
    ax.scatter(
        x[~sig],
        y[~sig],
        alpha=0.5,
        edgecolors="none",
        s=35,
        color="lightgray",
        zorder=2,
    )
    ax.scatter(
        x[sig], y[sig], alpha=0.7, edgecolors="none", s=35, color="steelblue", zorder=3
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)

    # Each count sits on its own side of the identity line (above-left vs.
    # below-right), clear of the q=0.05 reference lines.
    n_above = int(np.sum(y > x))
    n_below = int(np.sum(y < x))
    ax.text(
        0.12,
        0.96,
        f"above identity (val worse): {n_above}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="black",
    )
    ax.text(
        0.97,
        0.10,
        f"below identity (val better): {n_below}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        color="black",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _rug_missing_val(ax: plt.Axes, x_missing: np.ndarray) -> None:
    """Rug hypotheses whose val result is missing along the bottom x-axis.

    They have a known train x-position but no y (val) value, so they're drawn in
    a thin strip just below the data area rather than silently dropped.
    """
    if len(x_missing) == 0:
        return
    ax.set_ylim(-0.06, 1.0)
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.4)
    ax.scatter(
        x_missing,
        np.full(len(x_missing), -0.03),
        marker="|",
        s=140,
        linewidths=1.3,
        color=_MISSING_COLOR,
        clip_on=False,
        zorder=4,
        label=f"val missing (n={len(x_missing)})",
    )


def plot_pval_comparison(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot of train_p_value vs val_p_value with marginal histograms.

    Shows identity line, OLS fit (R², p-value), LOESS with bootstrap 95% CI,
    and histogram marginals for train (top) and val (right).
    df must have columns: train_p_value, val_p_value. Null rows are dropped.
    """
    paired = df["train_p_value"].notna() & df["val_p_value"].notna()
    missing = df["train_p_value"].notna() & df["val_p_value"].isna()
    plot_df = df[paired].copy()
    x_missing = df.loc[missing, "train_p_value"].astype(float).to_numpy()

    if len(plot_df) < _MIN_POINTS:
        _save_placeholder(
            output_path,
            f"Not enough paired p-values to plot "
            f"(n={len(plot_df)}; {len(x_missing)} with val missing)",
        )
        return

    x = plot_df["train_p_value"].astype(float).to_numpy()
    y = plot_df["val_p_value"].astype(float).to_numpy()

    fig, ax, ax_top, ax_right = _make_marginal_fig(x, y)
    _plot_scatter_with_fits(ax, x, y)
    _rug_missing_val(ax, x_missing)

    ax_top.set_ylabel("density", fontsize=8)
    ax_right.set_xlabel("density", fontsize=8, rotation=0, labelpad=4)
    ax.set_xlabel("train p-value")
    ax.set_ylabel("val p-value")
    ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_qval_comparison(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter of BH q-values (train v val) with marginal histograms and 0.05 lines.

    BH correction is applied independently to train and val p-values.
    df must have columns: train_p_value, val_p_value. Null rows are dropped.
    """
    plt.rcParams["font.family"] = "Arial"

    has_train = df["train_p_value"].notna()
    paired = has_train & df["val_p_value"].notna()
    missing = has_train & df["val_p_value"].isna()
    plot_df = df[paired].copy()

    if len(plot_df) < _MIN_POINTS:
        _save_placeholder(
            output_path,
            f"Not enough paired p-values to plot "
            f"(n={len(plot_df)}; {int(missing.sum())} with val missing)",
        )
        return

    # BH-correct train across every hypothesis with a train p-value (independent of
    # whether val ran), so missing-val rows still get a train q for placement.
    train_q_all = pd.Series(
        multipletests(
            df.loc[has_train, "train_p_value"].astype(float).to_numpy(),
            method="fdr_bh",
        )[1],
        index=df.index[has_train],
    )
    _, val_q, _, _ = multipletests(
        plot_df["val_p_value"].astype(float).to_numpy(), method="fdr_bh"
    )

    train_q = train_q_all.loc[plot_df.index].to_numpy()
    q_missing = train_q_all.loc[df.index[missing]].to_numpy()

    fig, ax, ax_top, ax_right = _make_marginal_fig(train_q, val_q)
    _plot_qval_scatter(ax, train_q, val_q)
    _rug_missing_val(ax, q_missing)

    ax.axvline(0.05, color="tab:red", linestyle=":", linewidth=1)
    ax.axhline(0.05, color="tab:red", linestyle=":", linewidth=1)

    ax_top.set_ylabel("density", fontsize=8)
    ax_right.set_xlabel("density", fontsize=8, rotation=0, labelpad=4)
    ax.set_xlabel("train q-value")
    ax.set_ylabel("val q-value")

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
