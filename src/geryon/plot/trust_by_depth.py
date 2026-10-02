"""Plot the critic's trustworthiness rating by refinement depth.

Per depth: a jittered stripplot of the individual ratings on the left, and a small
normalized histogram on the right, one horizontal bar per rating level whose length
is that level's share within the depth.
"""

import matplotlib.pyplot as plt
import numpy as np

from geryon.plot._critiques import (
    DEPTH_LABEL,
    GOOD,
    compute_depths,
    critiqued,
    load_hypotheses,
    parse_args,
)

_DOT_X = -0.18  # dots sit left of the depth tick
_BAR_LEFT = -0.02  # proportion bars start just right of the dots
_BAR_SCALE = 0.45  # depth-units a proportion of 1.0 spans
_BAR_H = 0.55  # bar thickness in rating-units


def main() -> None:
    args = parse_args("Plot trustworthiness rating by refinement depth")
    hyps = load_hypotheses(args.data_dir)
    depth = compute_depths(hyps)
    rows = critiqued(hyps, args.chain)

    groups: dict[int, list[int]] = {}
    for _, h, c in rows:
        groups.setdefault(depth[h.hypothesis_id], []).append(c.trustworthiness)
    all_depths = sorted(groups)

    fig, ax = plt.subplots(figsize=(8, 1.9))
    rng = np.random.default_rng(42)

    for d in all_depths:
        vals = groups[d]
        jitter = rng.uniform(-0.07, 0.07, size=len(vals))
        ax.scatter(d + _DOT_X + jitter, vals, color=GOOD, alpha=0.5, s=18, zorder=3)
        for rating in (1, 2, 3):
            prop = vals.count(rating) / len(vals)
            ax.barh(
                rating,
                prop * _BAR_SCALE,
                left=d + _BAR_LEFT,
                height=_BAR_H,
                color=GOOD,
                alpha=0.3,
                zorder=2,
            )
        ax.text(d, 3.45, f"N={len(vals)}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel(DEPTH_LABEL)
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
