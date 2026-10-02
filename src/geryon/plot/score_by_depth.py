"""Plot the summary score (novelty − confound + trust) by refinement depth."""

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._critiques import (
    DEPTH_LABEL,
    GOOD,
    SCORE_LABEL,
    compute_depths,
    critiqued,
    load_hypotheses,
    parse_args,
    score,
)


def main() -> None:
    args = parse_args("Plot summary score by refinement depth")
    hyps = load_hypotheses(args.data_dir)
    depth = compute_depths(hyps)
    rows = critiqued(hyps, args.chain)

    groups: dict[int, list[int]] = {}
    for _, h, c in rows:
        groups.setdefault(depth[h.hypothesis_id], []).append(score(c))
    all_depths = sorted(groups)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    rng = np.random.default_rng(42)

    if all_depths:
        ax.boxplot(
            [groups[d] for d in all_depths],
            positions=all_depths,
            widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": GOOD, "alpha": 0.3},
            medianprops={"color": GOOD, "linewidth": 2},
            whiskerprops={"color": "grey"},
            capprops={"color": "grey"},
            flierprops={"marker": ""},
        )
        for d in all_depths:
            vals = groups[d]
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(d + jitter, vals, color=GOOD, alpha=0.5, s=20, zorder=3)
            ax.text(d, 5.6, f"N={len(vals)}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(all_depths)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel(DEPTH_LABEL)
    ax.set_ylabel(SCORE_LABEL)
    ax.set_ylim(-2, 6)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
