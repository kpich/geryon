"""Plot N/U/T rating distributions by refinement depth."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._loader import load_hypotheses

# Okabe-Ito colorblind-safe palette; GOOD/NEUTRAL/BAD mapped per dimension
_GOOD = "#0072B2"  # blue
_NEUTRAL = "#E69F00"  # amber
_BAD = "#D55E00"  # vermillion

COLORS: dict[str, dict[int, str]] = {
    "novelty": {1: _BAD, 2: _NEUTRAL, 3: _GOOD},  # surprising=good
    "uncontrolled": {1: _GOOD, 2: _NEUTRAL, 3: _BAD},  # clean=good
    "trustworthiness": {1: _BAD, 2: _NEUTRAL, 3: _GOOD},  # credible=good
}

DIRECTION: dict[str, str] = {
    "novelty": "↑ higher = better",
    "uncontrolled": "↓ lower = better",
    "trustworthiness": "↑ higher = better",
}

DIMENSIONS = ["novelty", "uncontrolled", "trustworthiness"]
DIMENSION_LABELS = {
    "novelty": "Novelty",
    "uncontrolled": "Uncontrolled",
    "trustworthiness": "Trustworthiness",
}


def compute_depths(hyps: list) -> dict[str, int]:
    """Compute refinement depth for each hypothesis via BFS."""
    id_to_parent_raw: dict[str, str | None] = {
        h.hypothesis_id: h.proposal.refines_hypothesis for h in hyps
    }
    known_ids = set(id_to_parent_raw)

    # Resolve short IDs (8-char prefix) to full UUIDs for backward compat
    short_to_full = {hid[:8]: hid for hid in known_ids}
    id_to_parent: dict[str, str | None] = {}
    for hid, parent in id_to_parent_raw.items():
        if parent is not None and parent not in known_ids:
            parent = short_to_full.get(parent, parent)
        id_to_parent[hid] = parent

    depth: dict[str, int] = {}

    # Seed depth-0 nodes
    for hyp in hyps:
        parent = id_to_parent[hyp.hypothesis_id]
        if parent is None or parent not in known_ids:
            depth[hyp.hypothesis_id] = 0

    # BFS propagation
    changed = True
    while changed:
        changed = False
        for hyp in hyps:
            hid = hyp.hypothesis_id
            if hid in depth:
                continue
            parent = id_to_parent[hid]
            if parent in depth:
                depth[hid] = depth[parent] + 1
                changed = True

    # Anything unresolved (e.g. circular): treat as depth 0
    for hyp in hyps:
        depth.setdefault(hyp.hypothesis_id, 0)

    return depth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rating distributions by refinement depth"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hyps = load_hypotheses(args.data_dir)
    depth = compute_depths(hyps)

    # Filter to hypotheses rated on at least one dimension
    rated = [
        h
        for h in hyps
        if any(
            v is not None
            for v in (
                h.effective_rating.novelty,
                h.effective_rating.uncontrolled,
                h.effective_rating.trustworthiness,
            )
        )
    ]

    all_depths = sorted({depth[h.hypothesis_id] for h in rated})

    fig, axes = plt.subplots(3, 1, figsize=(10, 6))
    fig.suptitle("Rating Distributions by Refinement Depth", fontsize=14, y=1.01)

    for ax, dim in zip(axes, DIMENSIONS, strict=False):
        # Count ratings per depth level for this dimension
        depth_counts: dict[int, dict[int, int]] = defaultdict(
            lambda: {1: 0, 2: 0, 3: 0}
        )
        for hyp in rated:
            d = depth[hyp.hypothesis_id]
            rating = getattr(hyp.effective_rating, dim)
            if rating is not None:
                depth_counts[d][rating] += 1

        x_positions = list(range(len(all_depths)))
        bottoms = np.zeros(len(all_depths))

        for rating_val in (1, 2, 3):
            counts = np.array(
                [depth_counts[d][rating_val] for d in all_depths], dtype=float
            )
            totals = np.array(
                [sum(depth_counts[d].values()) for d in all_depths], dtype=float
            )
            proportions = np.where(totals > 0, counts / totals, 0.0)

            ax.bar(
                x_positions,
                proportions,
                bottom=bottoms,
                color=[COLORS[dim][rating_val]] * len(all_depths),
                label=str(rating_val),
                width=0.6,
            )
            bottoms += proportions

        # Annotate bars with N (dimension-specific count)
        for i, d in enumerate(all_depths):
            total = sum(depth_counts[d].values())
            if total > 0:
                ax.text(i, 1.02, f"N={total}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(d) for d in all_depths])
        ax.set_xlabel("Refinement depth (0 = original hypothesis)")
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{DIMENSION_LABELS[dim]} ({DIRECTION[dim]})")
        ax.legend(title="Rating", loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
