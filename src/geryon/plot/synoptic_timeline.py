"""Synoptic timeline: cost, tokens, ratings, and train/val q-values as columns
that all share one vertical hypothesis axis. Intended to grow — handpicked-
hypothesis callouts linking the panels will be added later.

Hypotheses run DOWN the shared y-axis (1 at top), in creation order. Each panel
is a column showing a different quantity on its x-axis, so a given hypothesis
lines up as a horizontal band across every panel. Cost/tokens are metered per
generation iteration, so they appear as staircases (the ~3 hypotheses from one
iteration share that iteration's cumulative value):

    hyp |  cost   | tokens  | ratings | q-value |
      1 |  ...    |  ...    |  ...    |  ...    |
      2 |  ...    |  ...    |  ...    |  ...    |
      v |         |         |         |         |
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from geryon.plot._loader import load_hypotheses
from geryon.plot.cost_over_time import (
    _INPUT_PRICE,
    _OUTPUT_PRICE,
    _estimated_cost,
    _input_volume,
    load_generation_usage,
)

# Okabe-Ito colorblind-safe palette.
_BLUE = "#0072B2"
_AMBER = "#E69F00"
_GREEN = "#009E73"
_VERMILLION = "#D55E00"

_LW = 0.9  # thin lines throughout

_RATING_DIMS = [
    ("novelty", "Novelty", _BLUE),
    ("uncontrolled", "Uncontrolled", _AMBER),
    ("trustworthiness", "Trustworthiness", _GREEN),
]


def _fmt_tokens(v: float, _pos: int) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:.0f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.0f}k"
    return f"{v:.0f}"


def _session_id(trace_path: str) -> str:
    """Session UUID is the directory holding the trace file."""
    return Path(trace_path).parent.name


def iteration_index(events: list[dict]) -> dict[tuple[str, int | None], int]:
    """Map (session_id, iteration) -> global 1-based iteration index.

    Events arrive ordered by timestamp, so the index is a single sequence across
    all sessions. Hypotheses are placed via the same key.
    """
    return {
        (_session_id(e["_session"]), e.get("iteration")): i
        for i, e in enumerate(events, 1)
    }


def _hyp_iteration(h, itmap: dict) -> int | None:
    return itmap.get((h.session_id, h.iteration))


def _pooled_cache_fractions(events: list[dict]) -> tuple[float, float] | None:
    """Pool the (cache_read, cache_write) fractions of input volume across ALL
    cached runs. Pooling (rather than using only the latest run) lets the
    with-caching estimate refine as more cached hypotheses are generated.
    """
    cached = [e for e in events if e.get("cache_read_tokens", 0) > 0]
    if not cached:
        return None
    volume = sum(_input_volume(e) for e in cached)
    if volume <= 0:
        return None
    f_read = sum(e["cache_read_tokens"] for e in cached) / volume
    f_write = sum(e.get("cache_creation_tokens", 0) for e in cached) / volume
    return f_read, f_write


def _uncached_cost(e: dict) -> float:
    """Cost of an event as if caching were off: full prompt volume at input rate."""
    return _input_volume(e) * _INPUT_PRICE + e.get("output_tokens", 0) * _OUTPUT_PRICE


def _iteration_cumulatives(
    events: list[dict],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Cumulative (without_cost, with_cost, input, output) per global iteration.

    with_cost is None when no run used caching. Indexed 0-based by global
    iteration order (i.e. element g-1 is the cumulative value through iteration g).
    """
    without = np.cumsum([_uncached_cost(e) for e in events])
    cum_in = np.cumsum([_input_volume(e) for e in events])
    cum_out = np.cumsum([e.get("output_tokens", 0) for e in events])
    fractions = _pooled_cache_fractions(events)
    with_c = (
        np.cumsum([_estimated_cost(e, fractions) for e in events])
        if fractions is not None
        else None
    )
    return without, with_c, cum_in, cum_out


def _per_hypothesis(arr: np.ndarray, gidx: list[int | None]) -> np.ndarray:
    """Expand a per-iteration array to one value per hypothesis via each
    hypothesis's global iteration index (NaN where unmapped)."""
    return np.array(
        [arr[g - 1] if g is not None else np.nan for g in gidx], dtype=float
    )


def _plot_cost(ax: plt.Axes, y, without, with_c) -> None:
    ax.plot(without, y, color=_VERMILLION, linewidth=_LW, label="without caching")
    if with_c is not None:
        ax.plot(with_c, y, color=_BLUE, linewidth=_LW, label="with caching")
    ax.set_xlabel("Cumulative cost (USD)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_tokens(ax: plt.Axes, y, cum_in, cum_out) -> None:
    ax.plot(cum_in, y, color=_BLUE, linewidth=_LW)
    ax.set_xlabel("Cumulative input", color=_BLUE)
    ax.tick_params(axis="x", labelcolor=_BLUE)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))
    ax.grid(True, alpha=0.3)

    ax_t = ax.twiny()  # second x-axis (top), shares the hypothesis y-axis
    ax_t.plot(cum_out, y, color=_VERMILLION, linewidth=_LW)
    ax_t.set_xlabel("Cumulative output", color=_VERMILLION)
    ax_t.tick_params(axis="x", labelcolor=_VERMILLION)
    ax_t.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))


def _plot_ratings(ax: plt.Axes, hyps: list, y: np.ndarray) -> None:
    for dim, label, color in _RATING_DIMS:
        pts = [
            (y[i], getattr(h.effective_rating, dim))
            for i, h in enumerate(hyps)
            if getattr(h.effective_rating, dim) is not None
        ]
        if not pts:
            continue
        yy = np.array([p[0] for p in pts], dtype=float)
        rr = np.array([p[1] for p in pts], dtype=float)
        ax.scatter(rr, yy, color=color, alpha=0.25, s=12, zorder=1)

        window = max(3, len(rr) // 5)
        if len(rr) >= window:
            roll = np.convolve(rr, np.ones(window) / window, mode="valid")
            half = window // 2
            ax.plot(
                roll,
                yy[half : half + len(roll)],
                color=color,
                linewidth=_LW,
                label=label,
                zorder=2,
            )
        else:
            ax.plot(rr, yy, color=color, linewidth=_LW, label=label, zorder=2)

    ax.set_xlabel("Rating")
    ax.set_xlim(0.7, 3.3)
    ax.set_xticks([1, 2, 3])
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_qvalues(ax: plt.Axes, input_csv: Path, idx_by_id: dict[str, int]) -> None:
    df = pd.read_csv(input_csv)
    df = df[df["val_success"].astype(str) == "True"]
    df = df[df["train_p_value"].notna() & df["val_p_value"].notna()].copy()
    if df.empty:
        ax.text(0.5, 0.5, "no val data", ha="center", va="center", fontsize=9)
        return

    _, train_q, _, _ = multipletests(
        df["train_p_value"].astype(float).to_numpy(), method="fdr_bh"
    )
    _, val_q, _, _ = multipletests(
        df["val_p_value"].astype(float).to_numpy(), method="fdr_bh"
    )
    df["train_q"], df["val_q"] = train_q, val_q
    df["hy"] = df["hypothesis_id"].map(idx_by_id)
    df = df[df["hy"].notna()].sort_values("hy")
    if df.empty:
        ax.text(0.5, 0.5, "no matching val data", ha="center", va="center", fontsize=9)
        return

    ax.scatter(df["train_q"], df["hy"], color=_AMBER, alpha=0.7, s=18, label="train q")
    ax.scatter(df["val_q"], df["hy"], color=_BLUE, alpha=0.7, s=18, label="val q")
    ax.axvline(0.05, color=_VERMILLION, linestyle=":", linewidth=_LW)

    ax.set_xlabel("q-value")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synoptic timeline of a run")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--input", type=Path, help="val_results.csv for the q-value panel (optional)"
    )
    args = parser.parse_args()

    events = load_generation_usage(args.data_dir)
    hyps = load_hypotheses(args.data_dir)  # chronological (created_at)
    itmap = iteration_index(events)

    # Shared axis is the hypothesis index; cost/tokens (metered per generation
    # iteration) are expanded to each hypothesis via its iteration.
    y = np.arange(1, len(hyps) + 1)
    gidx = [_hyp_iteration(h, itmap) for h in hyps]

    fig, (ax_cost, ax_tok, ax_rate, ax_q) = plt.subplots(
        1, 4, figsize=(15, 8), sharey=True
    )

    if hyps and events:
        without, with_c, cum_in, cum_out = _iteration_cumulatives(events)
        _plot_cost(
            ax_cost,
            y,
            _per_hypothesis(without, gidx),
            _per_hypothesis(with_c, gidx) if with_c is not None else None,
        )
        _plot_tokens(
            ax_tok, y, _per_hypothesis(cum_in, gidx), _per_hypothesis(cum_out, gidx)
        )
    if hyps:
        _plot_ratings(ax_rate, hyps, y)
    if args.input is not None and args.input.exists() and hyps:
        idx_by_id = {h.hypothesis_id: i + 1 for i, h in enumerate(hyps)}
        _plot_qvalues(ax_q, args.input, idx_by_id)

    ax_cost.set_ylabel("Hypothesis")
    if not ax_cost.yaxis_inverted():
        ax_cost.invert_yaxis()  # hypothesis 1 at the top (shared across all panels)

    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
