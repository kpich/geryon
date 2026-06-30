"""Plot trustworthiness rating by refinement depth.

Per depth: a jittered stripplot of the individual ratings on the left, and a small
normalized histogram on the right — one horizontal bar per rating level (1/2/3),
length proportional to that level's share within the depth.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geryon.legacy.plot._loader import load_hypotheses
from geryon.legacy.plot.ratings_by_depth import compute_depths

_BLUE = "#0072B2"
_DOT_X = -0.18  # dots sit left of the depth tick
_BAR_LEFT = -0.02  # proportion bars start just right of the dots
_BAR_SCALE = 0.45  # depth-units a proportion of 1.0 spans
_BAR_H = 0.55  # bar thickness in rating-units


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot trustworthiness rating by refinement depth"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hyps = load_hypotheses(args.data_dir)
    depth = compute_depths(hyps)

    scored = [(h, h.effective_rating.trustworthiness) for h in hyps]
    scored = [(h, t) for h, t in scored if t is not None]

    all_depths = sorted({depth[h.hypothesis_id] for h, _ in scored})
    groups = {
        d: [t for h, t in scored if depth[h.hypothesis_id] == d] for d in all_depths
    }

    fig, ax = plt.subplots(figsize=(8, 1.9))

    rng = np.random.default_rng(42)

    for d in all_depths:
        vals = groups[d]
        n = len(vals)

        jitter = rng.uniform(-0.07, 0.07, size=n)
        ax.scatter(
            [d + _DOT_X + j for j in jitter],
            vals,
            color=_BLUE,
            alpha=0.5,
            s=18,
            zorder=3,
        )

        for rating in (1, 2, 3):
            prop = sum(1 for v in vals if v == rating) / n if n else 0.0
            ax.barh(
                rating,
                prop * _BAR_SCALE,
                left=d + _BAR_LEFT,
                height=_BAR_H,
                color=_BLUE,
                alpha=0.3,
                zorder=2,
            )

        ax.text(d, 3.45, f"N={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Refinement depth (0 = original hypothesis)")
    ax.set_ylabel("Trustworthiness")
    ax.set_yticks([1, 2, 3])
    ax.set_ylim(0.5, 3.7)
    if all_depths:
        ax.set_xticks(all_depths)
        ax.set_xlim(min(all_depths) - 0.5, max(all_depths) + 0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
