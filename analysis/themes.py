"""Recurring themes across hypotheses, and how rarely the generator marks them as
refinements.

Themes are hand-written keyword rules over title + description, good enough to see
revisits; they are not a classifier. Every prior hypothesis is in the generator's
context (no window), so a revisit is a choice, not forgetting.

    uv run python analysis/themes.py   # -> plots/analysis/themes.pdf
"""

from pathlib import Path

import load
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "plots" / "analysis" / "themes.pdf"

THEMES = {
    "STK11/KEAP1/NRF2 in NSCLC": r"\bSTK11\b|\bKEAP1\b|NFE2L2",
    "Thyroid / TSH irAE": r"\bTSH\b|thyroid",
    "AR / PSA in prostate": r"\bAR (?:amp|L702)|PSA|androgen|ARSi",
    "CA19-9 non-secretor": r"CA19-9|CA 19-9|non-secretor",
    "9p21 / MTAP": r"MTAP|9p21|CDKN2A",
    "Antigen presentation (B2M/JAK)": r"\bB2M\b|JAK1",
    "Treatment-induced mutational scar": r"scar|hypermutation|signature|imprint",
    "Genetic ancestry": r"ancestry",
    "MMR / MSI": r"MSI|MMR",
}
GREY = "#999999"
BLUE = "#0072B2"
ORANGE = "#D55E00"


def main() -> None:
    h = load.hypotheses()
    text = h["title"] + " " + h["effect_size_type"].fillna("")
    order = list(THEMES)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for row, (_theme, pattern) in enumerate(THEMES.items()):
        hit = h[text.str.contains(pattern, regex=True)]
        ax.plot(hit.index, [row] * len(hit), color=GREY, lw=1, zorder=1)
        ax.scatter(hit.index, [row] * len(hit), s=40, color=BLUE, zorder=3)
    other = h[~text.str.contains("|".join(THEMES.values()), regex=True)]
    ax.scatter(other.index, [len(order)] * len(other), s=40, color=GREY, zorder=3)
    ids = dict(zip(h["hypothesis_id"], h.index, strict=False))
    for i, parent in h["refines"].dropna().items():
        ax.annotate(
            "refines",
            xy=(i, -0.6),
            xytext=(ids[parent], -0.6),
            arrowprops={"arrowstyle": "->", "color": ORANGE},
            color=ORANGE,
            fontsize=8,
            va="center",
        )
    starts = h.groupby("session_id").head(1).index
    for s in starts[1:]:
        ax.axvline(s - 0.5, color="#dddddd", lw=0.8, zorder=0)
    ax.set_yticks(range(len(order) + 1), order + ["(no recurring theme)"])
    ax.set_ylim(-1.1, len(order) + 0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Hypothesis, in creation order (vertical lines = session boundaries)")
    n_ref = h["refines"].notna().sum()
    ax.set_title(
        f"Themes recur across sessions; only {n_ref} of {len(h)} hypotheses set "
        "`refines`",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
