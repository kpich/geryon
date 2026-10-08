"""Plot the summary score (novelty − confound + trust) over the hypothesis sequence."""

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._critiques import (
    GOOD,
    SCORE_LABEL,
    critiqued,
    load_hypotheses,
    parse_args,
    plot_rolling_mean,
    score,
)
from geryon.plot._save import save


def main() -> None:
    args = parse_args("Plot summary score over the hypothesis sequence")
    rows = critiqued(load_hypotheses(args.data_dir), args.chain)

    fig, ax = plt.subplots(figsize=(10, 2.5))

    if rows:
        xs = np.array([i for i, _, _ in rows])
        ys = np.array([score(c) for _, _, c in rows])
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(ys))
        ax.scatter(xs, ys + jitter, color=GOOD, alpha=0.6, s=20, zorder=2)
        plot_rolling_mean(ax, xs, ys)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hypothesis sequence")
    ax.set_ylabel(SCORE_LABEL)
    ax.set_ylim(-2, 6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save(args.output)


if __name__ == "__main__":
    main()
