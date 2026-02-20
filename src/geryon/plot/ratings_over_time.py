"""Plot N/U/T rating distributions over the hypothesis sequence."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._loader import load_hypotheses

# Colors match the annotator UI (annotate.html)
COLORS: dict[str, dict[int, str]] = {
    "novelty": {1: "#9ca3af", 2: "#d97706", 3: "#059669"},
    "uncontrolled": {1: "#059669", 2: "#d97706", 3: "#dc2626"},
    "trustworthiness": {1: "#dc2626", 2: "#d97706", 3: "#059669"},
}

YTICK_LABELS: dict[str, dict[int, str]] = {
    "novelty": {1: "1 (known)", 2: "2", 3: "3 (surprising)"},
    "uncontrolled": {1: "1 (clean)", 2: "2", 3: "3 (confounded)"},
    "trustworthiness": {1: "1 (spurious)", 2: "2", 3: "3 (credible)"},
}

DIMENSIONS = ["novelty", "uncontrolled", "trustworthiness"]
DIMENSION_LABELS = {
    "novelty": "Novelty",
    "uncontrolled": "Uncontrolled",
    "trustworthiness": "Trustworthiness",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rating distributions over hypothesis sequence"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hyps = load_hypotheses(args.data_dir)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Rating Distributions Over Time", fontsize=14, y=1.01)

    rng = np.random.default_rng(42)

    for ax, dim in zip(axes, DIMENSIONS, strict=False):
        pairs: list[tuple[int, int]] = []
        for idx, hyp in enumerate(hyps):
            rating = getattr(hyp.effective_rating, dim)
            if rating is not None:
                pairs.append((idx, rating))

        n_rated = len(pairs)

        if pairs:
            xs = np.array([p[0] for p in pairs])
            ys = np.array([p[1] for p in pairs])
            y_jitter = rng.uniform(-0.1, 0.1, size=len(ys))

            for rating_val in (1, 2, 3):
                mask = ys == rating_val
                if mask.any():
                    ax.scatter(
                        xs[mask],
                        ys[mask] + y_jitter[mask],
                        c=COLORS[dim][rating_val],
                        alpha=0.6,
                        s=20,
                        zorder=2,
                    )

            window = max(3, len(ys) // 5)
            if len(ys) >= window:
                sort_idx = np.argsort(xs)
                ys_sorted = ys[sort_idx]
                xs_sorted = xs[sort_idx]
                rolling = np.convolve(
                    ys_sorted.astype(float), np.ones(window) / window, mode="valid"
                )
                # Center the rolling mean over its window
                half = window // 2
                rolling_xs = xs_sorted[half : half + len(rolling)]
                ax.plot(
                    rolling_xs,
                    rolling,
                    color="black",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=3,
                )

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(
            [YTICK_LABELS[dim][1], YTICK_LABELS[dim][2], YTICK_LABELS[dim][3]]
        )
        ax.set_ylim(0.5, 3.5)
        ax.set_xlabel("Hypothesis sequence")
        ax.set_ylabel(DIMENSION_LABELS[dim])
        ax.set_title(f"{DIMENSION_LABELS[dim]} — N={n_rated} rated")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
