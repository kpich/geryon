"""How much of the "Opus 5.5" batch was written by the Opus 4.8 refusal fallback.

uv run python analysis/fallback.py   # -> plots/analysis/fallback.pdf
"""

from pathlib import Path

import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEN = "#0072B2"
CRIT = "#E69F00"
OUT = Path(__file__).resolve().parent.parent / "plots" / "analysis" / "fallback.pdf"


def main() -> None:
    ai = load.calls()
    hyps = load.hypotheses()
    per_conv = (
        ai.groupby(["hypothesis_id", "phase"])
        .agg(fallback=("fallback", "mean"), created_at=("created_at", "first"))
        .reset_index()
    )
    first_iter = pd.Timestamp(per_conv["created_at"].min())

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2, 1.2, 1]}
    )

    ax = axes[0]
    for phase, color in [("generation", GEN), ("critic", CRIT)]:
        p = per_conv[per_conv["phase"] == phase].sort_values("created_at")
        x = np.arange(len(p))
        ax.scatter(
            x, p["fallback"], s=22, color=color, alpha=0.7, label=phase, zorder=3
        )
        roll = p["fallback"].rolling(10, min_periods=5).mean()
        ax.plot(x, roll, color=color, lw=2)
    ax.set_xlabel("Hypothesis, in creation order (all sessions)")
    ax.set_ylabel("Share of model replies from fallback (4.8)")
    ax.set_title("Per conversation; line = rolling mean of 10")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    gen = ai[ai["phase"] == "generation"].copy()
    gen["rel"] = gen["pos"] / (gen["n_calls"] - 1).clip(lower=1)
    bins = pd.cut(gen["rel"], np.linspace(-0.001, 1, 6))
    rate = gen.groupby(bins, observed=True)["fallback"].mean()
    centers = [b.mid for b in rate.index]
    ax.plot(centers, rate.values, color=GEN, lw=2, marker="o", ms=8)
    prev = gen.groupby("conv")["fallback"].shift()
    sticky = gen.loc[prev == True, "fallback"].mean()  # noqa: E712
    fresh = gen.loc[prev == False, "fallback"].mean()  # noqa: E712
    ax.text(
        0.03,
        0.97,
        f"P(fallback | previous reply fallback) = {sticky:.2f}\n"
        f"P(fallback | previous reply primary) = {fresh:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color="#333",
    )
    ax.set_xlabel("Position in generator conversation (0 = first reply)")
    ax.set_ylabel("Fallback rate")
    ax.set_title("Refusals grow as the conversation grows")
    ax.set_ylim(0, max(0.5, rate.max() * 1.15))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    crit = per_conv[per_conv["phase"] == "critic"].merge(hyps, on="hypothesis_id")
    crit["who"] = np.select(
        [crit["fallback"] == 0, crit["fallback"] == 1], ["all 5.5", "all 4.8"], "mixed"
    )
    order = ["all 5.5", "mixed", "all 4.8"]
    rng = np.random.default_rng(0)
    for i, who in enumerate(order):
        c = crit[crit["who"] == who]
        jitter = rng.uniform(-0.15, 0.15, len(c))
        ax.scatter(
            i + jitter,
            c["trust"] + rng.uniform(-0.1, 0.1, len(c)),
            s=22,
            color=CRIT,
            alpha=0.7,
            zorder=3,
        )
        ax.text(
            i,
            3.45,
            f"n={len(c)}\nholds_up {c['holds_up'].mean():.0%}",
            ha="center",
            fontsize=8.5,
            color="#333",
        )
    ax.set_xticks(range(3), order)
    ax.set_yticks([1, 2, 3])
    ax.set_ylim(0.6, 3.9)
    ax.set_xlabel("Who wrote the critique")
    ax.set_ylabel("Critic trustworthiness")
    ax.set_title("Critic scores by model")
    ax.grid(axis="y", alpha=0.3)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Opus 5.5 → 4.8 refusal fallback: "
        f"{per_conv['hypothesis_id'].nunique()} hypotheses with "
        f"per-call model logs, since {first_iter:%Y-%m-%d}",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
