"""Shared loading and helpers for the plots of critic ratings.

Hypotheses are ordered by creation time across every session under ``data_dir``.
Refinement depth is computed over all chains before any chain filter, so a hypothesis
whose parent sits in another chain keeps its true depth.
"""

import argparse
from pathlib import Path

from matplotlib.axes import Axes
import numpy as np

from geryon.codeflow.context import load_prior_hypotheses
from geryon.codeflow.models import CodeCritique, CodeHypothesis

# Okabe-Ito colorblind-safe palette; GOOD/NEUTRAL/BAD mapped per dimension.
GOOD = "#0072B2"
NEUTRAL = "#E69F00"
BAD = "#D55E00"

DIMENSIONS = ["novelty", "confound_risk", "trustworthiness"]
DIMENSION_LABELS = {
    "novelty": "Novelty",
    "confound_risk": "Confound risk",
    "trustworthiness": "Trustworthiness",
}
COLORS: dict[str, dict[int, str]] = {
    "novelty": {1: BAD, 2: NEUTRAL, 3: GOOD},
    "confound_risk": {1: GOOD, 2: NEUTRAL, 3: BAD},
    "trustworthiness": {1: BAD, 2: NEUTRAL, 3: GOOD},
}
YTICK_LABELS: dict[str, dict[int, str]] = {
    "novelty": {1: "1 (trivial)", 2: "2", 3: "3 (novel)"},
    "confound_risk": {1: "1 (low)", 2: "2", 3: "3 (high)"},
    "trustworthiness": {1: "1 (weak)", 2: "2", 3: "3 (solid)"},
}
SCORE_LABEL = "Score  (novelty − confound + trust)"
DEPTH_LABEL = "Refinement depth (0 = original hypothesis)"


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir", required=True, type=Path, help="Searched for hypotheses.jsonl"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chain", default=None, help="Only this chain (default: all)")
    return parser.parse_args()


def load_hypotheses(data_dir: Path) -> list[CodeHypothesis]:
    hyps = load_prior_hypotheses(data_dir, current_session_id="")
    return sorted(hyps, key=lambda h: (h.created_at, h.iteration or 0))


def compute_depths(hyps: list[CodeHypothesis]) -> dict[str, int]:
    """Number of ``refines`` hops from each hypothesis back to a root."""
    parent = {h.hypothesis_id: h.refines for h in hyps}
    missing = {p for p in parent.values() if p is not None and p not in parent}
    if missing:
        raise ValueError(
            f"{len(missing)} refined parent(s) not found under the data dir, "
            f"e.g. {sorted(missing)[0]}; point --data-dir at all the sessions"
        )
    depth: dict[str, int] = {}

    def resolve(hid: str) -> None:
        chain: list[str] = []
        cur: str | None = hid
        while cur is not None and cur not in depth:
            if cur in chain:
                raise ValueError(f"refinement cycle through {cur}")
            chain.append(cur)
            cur = parent[cur]
        base = -1 if cur is None else depth[cur]
        for i, h in enumerate(reversed(chain), start=1):
            depth[h] = base + i

    for h in hyps:
        resolve(h.hypothesis_id)
    return depth


def score(c: CodeCritique) -> int:
    return c.novelty - c.confound_risk + c.trustworthiness


def critiqued(
    hyps: list[CodeHypothesis], chain: str | None
) -> list[tuple[int, CodeHypothesis, CodeCritique]]:
    """(sequence index, hypothesis, critique) for every critiqued hypothesis.

    The index counts all hypotheses in the selected chain, so gaps show where an
    uncritiqued one fell.
    """
    selected = [h for h in hyps if chain is None or h.chain == chain]
    return [(i, h, h.critique) for i, h in enumerate(selected) if h.critique]


def plot_rolling_mean(ax: Axes, xs: np.ndarray, ys: np.ndarray) -> None:
    window = max(3, len(ys) // 5)
    if len(ys) < window:
        return
    order = np.argsort(xs)
    rolling = np.convolve(
        ys[order].astype(float), np.ones(window) / window, mode="valid"
    )
    half = window // 2
    ax.plot(
        xs[order][half : half + len(rolling)],
        rolling,
        color="black",
        linewidth=1.5,
        alpha=0.7,
        zorder=3,
        label=f"rolling mean (w={window})",
    )
    ax.legend(loc="upper right", fontsize=9)
