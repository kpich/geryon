"""Plot summary score (novelty − uncontrolled + trustworthiness) over hyp sequence."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._loader import load_hypotheses


def compute_score(rating) -> int | None:
    n, u, t = rating.novelty, rating.uncontrolled, rating.trustworthiness
    if n is None or u is None or t is None:
        return None
    return n - u + t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot summary score over hypothesis sequence"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    hyps = load_hypotheses(args.data_dir)

    pairs: list[tuple[int, int]] = []
    for idx, hyp in enumerate(hyps):
        score = compute_score(hyp.effective_rating)
        if score is not None:
            pairs.append((idx, score))

    fig, ax = plt.subplots(figsize=(10, 2.5))

    rng = np.random.default_rng(42)

    if pairs:
        xs = np.array([p[0] for p in pairs])
        ys = np.array([p[1] for p in pairs])

        y_jitter = rng.uniform(-0.1, 0.1, size=len(ys))
        ax.scatter(xs, ys + y_jitter, color="#0072B2", alpha=0.6, s=20, zorder=2)

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

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hypothesis sequence")
    ax.set_ylabel("Score  (novelty − uncontrolled + trustworthiness)")
    ax.set_ylim(-2, 6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
