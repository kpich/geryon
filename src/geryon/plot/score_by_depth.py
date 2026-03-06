"""Plot summary score (novelty − uncontrolled + trustworthiness) by refinement depth."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._loader import load_hypotheses
from geryon.plot.ratings_by_depth import compute_depths


def compute_score(rating) -> int | None:
    n, u, t = rating.novelty, rating.uncontrolled, rating.trustworthiness
    if n is None or u is None or t is None:
        return None
    return n - u + t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot summary score by refinement depth"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hyps = load_hypotheses(args.data_dir)
    depth = compute_depths(hyps)

    scored = [(h, compute_score(h.effective_rating)) for h in hyps]
    scored = [(h, s) for h, s in scored if s is not None]

    all_depths = sorted({depth[h.hypothesis_id] for h, _ in scored})
    groups = {
        d: [s for h, s in scored if depth[h.hypothesis_id] == d and s is not None]
        for d in all_depths
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
            ax.text(d, 5.6, f"N={len(vals)}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Refinement depth (0 = original hypothesis)")
    ax.set_ylabel("Score  (novelty − uncontrolled + trustworthiness)")
    ax.set_title("Hypothesis Score by Refinement Depth")
    ax.set_xticks(all_depths)
    ax.set_ylim(-2, 6)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
