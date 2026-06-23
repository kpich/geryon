"""Synoptic timeline: cost, tokens, ratings, and train/val q-values as columns
that all share one vertical hypothesis axis. Intended to grow — handpicked-
hypothesis callouts linking the panels will be added later.

Hypotheses run DOWN the shared y-axis (1 at top), in creation order. Each panel
is a column showing a different quantity on its x-axis, so a given hypothesis
lines up as a horizontal band across every panel. Cost/tokens are metered per
generation iteration, so they appear as staircases (the ~3 hypotheses from one
iteration share that iteration's cumulative value):

    hyp |  cost   | tokens  | ratings | q-value | lineage |
      1 |  ...    |  ...    |  ...    |  ...    |  ...    |
      2 |  ...    |  ...    |  ...    |  ...    |  ...    |
      v |         |         |         |         |         |

The lineage panel plots each hypothesis at (refinement depth, index) and links it
to the parent it refines, so refinement chains read left-to-right.
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
from geryon.plot.ratings_by_depth import compute_depths

# Okabe-Ito colorblind-safe palette.
_BLUE = "#0072B2"
_AMBER = "#E69F00"
_GREEN = "#009E73"
_VERMILLION = "#D55E00"

# Per-panel line weights: context panels are de-emphasized, the data panels
# (ratings / q-values) carry the eye, and the refinement tree stays light.
_LW_COST = 0.8  # cost & token staircases — thinnest
_LW_DATA = 1.9  # ratings & q-values — fat, the focus
_LW_TREE = 0.8  # parent/child refinement connectors — thin

# x-position of the off-scale gutter where missing-val hyps are marked (right of q=1).
_MISSING_GUTTER = 1.12

# q-value significance threshold; non-significant points are drawn at low alpha.
_SIG_Q = 0.05
_SIG_ALPHA = 0.9
_NONSIG_ALPHA = 0.25


def _legend_below(ax: plt.Axes, **kw) -> None:
    """Place the legend below the panel so it never covers the data."""
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=7,
        framealpha=0.95,
        **kw,
    )


# (dim, label, color, better_dir): better_dir is +1 if a higher rating is better
# (arrow points right) or -1 if lower is better (arrow points left). "Uncontrolled"
# is reversed: 1=clean is good, 3=confounded is bad.
_RATING_DIMS = [
    ("novelty", "Novelty", _BLUE, +1),
    ("uncontrolled", "Uncontrolled", _AMBER, -1),
    ("trustworthiness", "Trustworthiness", _GREEN, +1),
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
    ax.plot(without, y, color=_VERMILLION, linewidth=_LW_COST, label="without caching")
    if with_c is not None:
        ax.plot(with_c, y, color=_BLUE, linewidth=_LW_COST, label="with caching")
    ax.set_xlabel("Cumulative cost (USD)")
    _legend_below(ax)
    ax.grid(True, alpha=0.3)


def _plot_tokens(ax: plt.Axes, y, cum_in, cum_out) -> None:
    ax.plot(cum_in, y, color=_BLUE, linewidth=_LW_COST)
    ax.set_xlabel("Cumulative input", color=_BLUE)
    ax.tick_params(axis="x", labelcolor=_BLUE)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))
    ax.grid(True, alpha=0.3)

    ax_t = ax.twiny()  # second x-axis (top), shares the hypothesis y-axis
    ax_t.plot(cum_out, y, color=_VERMILLION, linewidth=_LW_COST)
    ax_t.set_xlabel("Cumulative output", color=_VERMILLION)
    ax_t.tick_params(axis="x", labelcolor=_VERMILLION)
    ax_t.xaxis.set_major_formatter(FuncFormatter(_fmt_tokens))


def _plot_ratings(ax: plt.Axes, hyps: list, y: np.ndarray) -> None:
    for dim, label, color, better_dir in _RATING_DIMS:
        # Direction of "better" differs per dimension (uncontrolled is reversed),
        # so it rides in the legend label rather than a single shared arrow.
        legend_label = (
            f"{label} (better →)" if better_dir > 0 else f"{label} (← better)"
        )
        pts = [
            (y[i], getattr(h.effective_rating, dim))
            for i, h in enumerate(hyps)
            if getattr(h.effective_rating, dim) is not None
        ]
        if not pts:
            continue
        yy = np.array([p[0] for p in pts], dtype=float)
        rr = np.array([p[1] for p in pts], dtype=float)
        ax.scatter(rr, yy, color=color, alpha=0.55, s=16, zorder=1)

        window = max(3, len(rr) // 5)
        if len(rr) >= window:
            roll = np.convolve(rr, np.ones(window) / window, mode="valid")
            half = window // 2
            ax.plot(
                roll,
                yy[half : half + len(roll)],
                color=color,
                linewidth=_LW_DATA,
                label=legend_label,
                zorder=2,
            )
        else:
            ax.plot(
                rr, yy, color=color, linewidth=_LW_DATA, label=legend_label, zorder=2
            )

    ax.set_xlabel("Rating")
    ax.set_xlim(0.7, 3.3)
    ax.set_xticks([1, 2, 3])
    _legend_below(ax)
    ax.grid(True, alpha=0.3)


def _sig_q_scatter(ax: plt.Axes, q, hy, color: str, label: str) -> None:
    """Scatter q-values, fading the non-significant ones (q > 0.05) to low alpha
    so the significant points carry the eye."""
    q = np.asarray(q, dtype=float)
    alphas = np.where(q <= _SIG_Q, _SIG_ALPHA, _NONSIG_ALPHA)
    ax.scatter(q, hy, color=color, alpha=alphas, s=34, label=label, zorder=3)


def _plot_qvalues(ax: plt.Axes, input_csv: Path, idx_by_id: dict[str, int]) -> None:
    raw = pd.read_csv(input_csv)
    raw["hy"] = raw["hypothesis_id"].map(idx_by_id)
    raw = raw[raw["hy"].notna()].copy()

    val_ok = (raw["val_success"].astype(str) == "True") & raw["val_p_value"].notna()
    has_val = raw[val_ok].copy()
    has_train = raw[raw["train_p_value"].notna()].copy()
    # Any mapped hyp without a usable val q-value, regardless of whether train ran.
    no_val = raw[~val_ok]

    if has_val.empty and has_train.empty and no_val.empty:
        ax.text(0.5, 0.5, "no val data", ha="center", va="center", fontsize=9)
        return

    # Train and val q-values are BH-corrected over their own available sets, so a
    # hyp can carry one without the other (train ran but val failed, or vice versa).
    if not has_train.empty:
        has_train["train_q"] = multipletests(
            has_train["train_p_value"].astype(float).to_numpy(), method="fdr_bh"
        )[1]
        _sig_q_scatter(ax, has_train["train_q"], has_train["hy"], _AMBER, "train q")
    if not has_val.empty:
        has_val["val_q"] = multipletests(
            has_val["val_p_value"].astype(float).to_numpy(), method="fdr_bh"
        )[1]
        _sig_q_scatter(ax, has_val["val_q"], has_val["hy"], _BLUE, "val q")
    if not has_train.empty or not has_val.empty:
        ax.axvline(_SIG_Q, color=_VERMILLION, linestyle=":", linewidth=_LW_DATA)

    # No-val hyps go in a gutter OFF the q-scale (right of q=1), not at a real q
    # position where a marker would read as a low/high q-value.
    if not no_val.empty:
        ax.axvline(_MISSING_GUTTER - 0.04, color="gray", linewidth=0.8, alpha=0.6)
        ax.scatter(
            np.full(len(no_val), _MISSING_GUTTER),
            no_val["hy"],
            marker="x",
            color=_VERMILLION,
            s=40,
            linewidths=1.6,
            clip_on=False,
            zorder=4,
            label=f"no val (n={len(no_val)})",
        )
        ax.text(
            _MISSING_GUTTER,
            1.01,
            "no val",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6,
            color=_VERMILLION,
        )

    ax.set_xlabel("q-value")
    # Pin ticks to the q-scale so the gutter is visibly off-scale, not a q value.
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlim(-0.02, _MISSING_GUTTER + 0.04 if not no_val.empty else 1.02)
    _legend_below(ax)
    ax.grid(True, alpha=0.3)


def _resolved_parents(hyps: list) -> dict[str, str | None]:
    """Map each hypothesis_id to its parent's full id within this set (or None).

    Mirrors compute_depths' short-id (8-char prefix) resolution so a parent recorded
    as a short prefix still links to the full hypothesis.
    """
    ids = {h.hypothesis_id for h in hyps}
    short_to_full = {hid[:8]: hid for hid in ids}
    parents: dict[str, str | None] = {}
    for h in hyps:
        parent = h.proposal.refines_hypothesis
        if parent is not None and parent not in ids:
            parent = short_to_full.get(parent, parent)
        parents[h.hypothesis_id] = parent if parent in ids else None
    return parents


def _plot_lineage(ax: plt.Axes, hyps: list, y: np.ndarray) -> None:
    """Refinement tree: x = refinement depth, y = hypothesis index (shared).

    Each hypothesis is a node at (depth, index); a thin connector links it to the
    parent it refines, so refinement chains read left-to-right while staying on
    their creation-order rows.
    """
    depth = compute_depths(hyps)
    parents = _resolved_parents(hyps)
    y_of = {h.hypothesis_id: y[i] for i, h in enumerate(hyps)}

    for h in hyps:
        parent = parents[h.hypothesis_id]
        if parent is None:
            continue
        ax.plot(
            [depth[parent], depth[h.hypothesis_id]],
            [y_of[parent], y_of[h.hypothesis_id]],
            color="gray",
            linewidth=_LW_TREE,
            alpha=0.7,
            zorder=1,
        )

    depths = np.array([depth[h.hypothesis_id] for h in hyps], dtype=float)
    is_root = np.array([parents[h.hypothesis_id] is None for h in hyps])
    ax.scatter(
        depths[is_root],
        y[is_root],
        color=_GREEN,
        s=22,
        zorder=2,
        label=f"original (n={int(is_root.sum())})",
    )
    if (~is_root).any():
        ax.scatter(
            depths[~is_root],
            y[~is_root],
            color=_BLUE,
            s=22,
            zorder=2,
            label=f"refinement (n={int((~is_root).sum())})",
        )

    max_d = int(depths.max()) if len(depths) else 0
    ax.set_xlabel("Refinement depth")
    ax.set_xlim(-0.4, max_d + 0.4)
    ax.set_xticks(range(max_d + 1))
    _legend_below(ax)
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

    fig, (ax_cost, ax_tok, ax_rate, ax_q, ax_tree) = plt.subplots(
        1, 5, figsize=(18, 8), sharey=True
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
        _plot_lineage(ax_tree, hyps, y)
    if args.input is not None and args.input.exists() and hyps:
        idx_by_id = {h.hypothesis_id: i + 1 for i, h in enumerate(hyps)}
        _plot_qvalues(ax_q, args.input, idx_by_id)

    ax_cost.set_ylabel("Hypothesis")
    if not ax_cost.yaxis_inverted():
        ax_cost.invert_yaxis()  # hypothesis 1 at the top (shared across all panels)

    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
