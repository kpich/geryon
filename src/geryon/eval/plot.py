"""Plotting utilities for hypothesis evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests

_N_BOOT = 500
_LOESS_FRAC = 0.75


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


def _make_marginal_fig(
    x: np.ndarray, y: np.ndarray
) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes]:
    """Create a figure with marginal KDE axes.

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

    grid = np.linspace(0.0, 1.0, 300)
    color = "steelblue"
    for vals, axis, horizontal in [(x, ax_top, True), (y, ax_right, False)]:
        try:
            d = gaussian_kde(vals)(grid)
        except Exception:
            continue
        if horizontal:
            axis.fill_between(grid, d, alpha=0.25, color=color)
            axis.plot(grid, d, color=color, linewidth=0.9)
        else:
            axis.fill_betweenx(grid, d, alpha=0.25, color=color)
            axis.plot(d, grid, color=color, linewidth=0.9)

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


def plot_pval_comparison(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot of train_p_value vs val_p_value with marginal densities.

    Shows identity line, OLS fit (R², p-value), LOESS with bootstrap 95% CI,
    and KDE marginals for train (top) and val (right).
    df must have columns: train_p_value, val_p_value. Null rows are dropped.
    """
    mask = df["train_p_value"].notna() & df["val_p_value"].notna()
    plot_df = df[mask].copy()

    x = plot_df["train_p_value"].astype(float).to_numpy()
    y = plot_df["val_p_value"].astype(float).to_numpy()

    fig, ax, ax_top, ax_right = _make_marginal_fig(x, y)
    _plot_scatter_with_fits(ax, x, y)

    ax_top.set_ylabel("density", fontsize=8)
    ax_right.set_xlabel("density", fontsize=8, rotation=0, labelpad=4)
    ax.set_xlabel("train p-value")
    ax.set_ylabel("val p-value")
    ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_qval_comparison(df: pd.DataFrame, output_path: str | Path) -> None:
    """Scatter plot of BH q-values (train v val) with marginal densities and 0.05 lines.

    BH correction is applied independently to train and val p-values.
    df must have columns: train_p_value, val_p_value. Null rows are dropped.
    """
    mask = df["train_p_value"].notna() & df["val_p_value"].notna()
    plot_df = df[mask].copy()

    train_p = plot_df["train_p_value"].astype(float).to_numpy()
    val_p = plot_df["val_p_value"].astype(float).to_numpy()

    _, train_q, _, _ = multipletests(train_p, method="fdr_bh")
    _, val_q, _, _ = multipletests(val_p, method="fdr_bh")

    fig, ax, ax_top, ax_right = _make_marginal_fig(train_q, val_q)
    _plot_scatter_with_fits(ax, train_q, val_q)

    ax.axvline(0.05, color="tab:red", linestyle=":", linewidth=1, label="q=0.05")
    ax.axhline(0.05, color="tab:red", linestyle=":", linewidth=1)

    ax_top.set_ylabel("density", fontsize=8)
    ax_right.set_xlabel("density", fontsize=8, rotation=0, labelpad=4)
    ax.set_xlabel("train q-value")
    ax.set_ylabel("val q-value")
    ax.legend(fontsize=9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
