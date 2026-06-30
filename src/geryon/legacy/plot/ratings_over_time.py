"""Plot N/U/T rating distributions over the hypothesis sequence."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geryon.legacy.plot._loader import load_hypotheses

# Okabe-Ito colorblind-safe palette; GOOD/NEUTRAL/BAD mapped per dimension
_GOOD = "#0072B2"  # blue
_NEUTRAL = "#E69F00"  # amber
_BAD = "#D55E00"  # vermillion

COLORS: dict[str, dict[int, str]] = {
    "novelty": {1: _BAD, 2: _NEUTRAL, 3: _GOOD},  # surprising=good
    "uncontrolled": {1: _GOOD, 2: _NEUTRAL, 3: _BAD},  # clean=good
    "trustworthiness": {1: _BAD, 2: _NEUTRAL, 3: _GOOD},  # credible=good
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

Y_LO, Y_HI = 0.5, 3.5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot rating distributions over hypothesis sequence"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hyps = load_hypotheses(args.data_dir)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6))

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

            # Background: time-bucketed proportion bars
            n_bins = max(4, min(20, n_rated // 6))
            x_min, x_max = int(xs.min()), int(xs.max()) + 1
            bin_edges = np.linspace(x_min, x_max, n_bins + 1)
            bin_width = bin_edges[1] - bin_edges[0]
            y_span = Y_HI - Y_LO

            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                bin_ys = [p[1] for p in pairs if lo <= p[0] < hi]
                if not bin_ys:
                    continue
                total = len(bin_ys)
                bottom = Y_LO
                for rv in (1, 2, 3):
                    prop = bin_ys.count(rv) / total
                    ax.bar(
                        (lo + hi) / 2,
                        prop * y_span,
                        bottom=bottom,
                        width=bin_width,
                        color=COLORS[dim][rv],
                        alpha=0.2,
                        zorder=0,
                    )
                    bottom += prop * y_span

            # Scatter points
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

            # Rolling mean
            window = max(3, len(ys) // 5)
            if len(ys) >= window:
                sort_idx = np.argsort(xs)
                ys_sorted = ys[sort_idx]
                xs_sorted = xs[sort_idx]
                rolling = np.convolve(
                    ys_sorted.astype(float), np.ones(window) / window, mode="valid"
                )
                half = window // 2
                rolling_xs = xs_sorted[half : half + len(rolling)]
                ax.plot(
                    rolling_xs,
                    rolling,
                    color="black",
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=3,
                    label=f"rolling mean (w={window})",
                )
                ax.legend(loc="upper right", fontsize=9)

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(
            [YTICK_LABELS[dim][1], YTICK_LABELS[dim][2], YTICK_LABELS[dim][3]]
        )
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_xlabel("Hypothesis sequence")
        ax.set_ylabel(DIMENSION_LABELS[dim])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
