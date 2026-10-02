"""Plot the critic's novelty / confound / trust rating mix by refinement depth."""

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._critiques import (
    COLORS,
    DEPTH_LABEL,
    DIMENSION_LABELS,
    DIMENSIONS,
    compute_depths,
    critiqued,
    load_hypotheses,
    parse_args,
)


def main() -> None:
    args = parse_args("Plot critic rating mix by refinement depth")
    hyps = load_hypotheses(args.data_dir)
    depth = compute_depths(hyps)
    rows = critiqued(hyps, args.chain)

    all_depths = sorted({depth[h.hypothesis_id] for _, h, _ in rows})
    x_positions = np.arange(len(all_depths))

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    for ax, dim in zip(axes, DIMENSIONS, strict=True):
        counts = {d: {1: 0, 2: 0, 3: 0} for d in all_depths}
        for _, h, c in rows:
            counts[depth[h.hypothesis_id]][getattr(c, dim)] += 1
        totals = np.array([sum(counts[d].values()) for d in all_depths], dtype=float)

        bottoms = np.zeros(len(all_depths))
        for rv in (1, 2, 3):
            props = np.array([counts[d][rv] for d in all_depths]) / totals
            ax.bar(
                x_positions,
                props,
                bottom=bottoms,
                color=COLORS[dim][rv],
                label=str(rv),
                width=0.6,
            )
            bottoms += props

        for x, n in zip(x_positions, totals, strict=True):
            ax.text(x, 1.02, f"N={n:.0f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel(DIMENSION_LABELS[dim])
        ax.set_ylim(0, 1.15)
        ax.legend(title="Rating", loc="upper right", fontsize=8, title_fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels([str(d) for d in all_depths])
    axes[-1].set_xlabel(DEPTH_LABEL)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
