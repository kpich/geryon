"""Plot the critic's novelty / confound / trust ratings over the hypothesis sequence."""

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._critiques import (
    COLORS,
    DIMENSION_LABELS,
    DIMENSIONS,
    YTICK_LABELS,
    critiqued,
    load_hypotheses,
    parse_args,
    plot_rolling_mean,
)
from geryon.plot._save import save

Y_LO, Y_HI = 0.5, 3.5


def main() -> None:
    args = parse_args("Plot critic ratings over the hypothesis sequence")
    rows = critiqued(load_hypotheses(args.data_dir), args.chain)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    rng = np.random.default_rng(42)

    for ax, dim in zip(axes, DIMENSIONS, strict=True):
        if rows:
            xs = np.array([i for i, _, _ in rows])
            ys = np.array([getattr(c, dim) for _, _, c in rows])

            # Background: time-bucketed proportion bars
            n_bins = max(4, min(20, len(rows) // 6))
            edges = np.linspace(xs.min(), xs.max() + 1, n_bins + 1)
            width = edges[1] - edges[0]
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                in_bin = ys[(xs >= lo) & (xs < hi)]
                if not len(in_bin):
                    continue
                bottom = Y_LO
                for rv in (1, 2, 3):
                    h = (in_bin == rv).mean() * (Y_HI - Y_LO)
                    ax.bar(
                        (lo + hi) / 2,
                        h,
                        bottom=bottom,
                        width=width,
                        color=COLORS[dim][rv],
                        alpha=0.2,
                        zorder=0,
                    )
                    bottom += h

            jitter = rng.uniform(-0.1, 0.1, size=len(ys))
            for rv in (1, 2, 3):
                mask = ys == rv
                ax.scatter(
                    xs[mask],
                    ys[mask] + jitter[mask],
                    c=COLORS[dim][rv],
                    alpha=0.6,
                    s=20,
                    zorder=2,
                )
            plot_rolling_mean(ax, xs, ys)

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels([YTICK_LABELS[dim][v] for v in (1, 2, 3)])
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_ylabel(DIMENSION_LABELS[dim])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Hypothesis sequence")

    plt.tight_layout()
    save(args.output)


if __name__ == "__main__":
    main()
