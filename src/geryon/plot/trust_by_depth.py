"""Plot trustworthiness rating by refinement depth."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._loader import load_hypotheses
from geryon.plot.ratings_by_depth import compute_depths


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

    fig, ax = plt.subplots(figsize=(8, 2.5))

    rng = np.random.default_rng(42)

    if all_depths:
        ax.boxplot(
            [groups[d] for d in all_depths],
            positions=all_depths,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": "#0072B2", "alpha": 0.3},
            medianprops={"color": "#0072B2", "linewidth": 2},
            whiskerprops={"color": "grey"},
            capprops={"color": "grey"},
            flierprops={"marker": ""},
        )

        for d in all_depths:
            vals = groups[d]
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                [d + j for j in jitter],
                vals,
                color="#0072B2",
                alpha=0.5,
                s=20,
                zorder=3,
            )
            ax.text(d, 3.4, f"N={len(vals)}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Refinement depth (0 = original hypothesis)")
    ax.set_ylabel("Trustworthiness")
    ax.set_xticks(all_depths)
    ax.set_yticks([1, 2, 3])
    ax.set_ylim(0.5, 3.7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
