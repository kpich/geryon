"""Do the generator's significant findings replicate on held-out patients as often as
they should?

The validation split is a quarter the size of explore, so even a true effect at
explore z=2.5 replicates (p<0.05, same sign) only ~20% of the time. Raw replication
rates therefore say little. Instead each hypothesis gets the replication probability
it would have if its explore estimate were the truth, P(z_val > 1.96) with
z_val ~ N(z_explore · se_explore/se_val, 1), and that is compared with what happened.

Left: validation z against explore z. The dashed line is "explore estimate is
true" (slope = median SE ratio), the solid one the fit through the origin; the gap
between them is the average inflation of the explore effects.
Right: expected vs observed replications among explore-significant hypotheses,
split by how many scripts the generator ran before submitting.

    uv run python analysis/replication.py   # -> plots/analysis/replication.pdf
"""

from pathlib import Path

import load
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).resolve().parent.parent / "plots" / "analysis" / "replication.pdf"
FEW = "#0072B2"
MANY = "#D55E00"
UNKNOWN = "#999999"
SPLIT = 12  # generator run_python calls; roughly the median


def main() -> None:
    t = load.holdout()
    # Effect direction is arbitrary (which arm is the reference), so fold every
    # hypothesis onto its own explore direction.
    sign = np.sign(t["z_explore"])
    t["z_explore"], t["z_val"] = t["z_explore"].abs(), t["z_val"] * sign
    k = float((t["se_explore"] / t["se_val"]).median())
    slope = float((t["z_explore"] * t["z_val"]).sum() / (t["z_explore"] ** 2).sum())
    t["group"] = np.where(
        t["run_python"].isna(),
        "unknown",
        np.where(t["run_python"] <= SPLIT, "few", "many"),
    )
    colors = {"few": FEW, "many": MANY, "unknown": UNKNOWN}
    labels = {
        "few": f"≤{SPLIT} scripts before submit",
        "many": f">{SPLIT} scripts before submit",
        "unknown": "not logged",
    }

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    for g in ["unknown", "few", "many"]:
        s = t[t["group"] == g]
        ax.scatter(
            s["z_explore"],
            s["z_val"],
            s=36,
            color=colors[g],
            label=labels[g],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    lim = float(np.ceil(t["z_explore"].max()))
    xs = np.array([0, lim])
    ax.plot(
        xs,
        k * xs,
        ls="--",
        color="#333",
        lw=1.2,
        label=f"if explore were true (slope {k:.2f})",
    )
    ax.plot(
        xs,
        slope * xs,
        color="#333",
        lw=1.2,
        label=f"fit (slope {slope:.2f}, {1 - slope / k:.0%} inflation)",
    )
    ax.axvspan(1.96, 3, color="#f3f3f3", zorder=0)
    ax.axvline(1.96, color="#ccc", lw=0.8, zorder=0)
    ax.axhline(1.96, color="#ccc", lw=0.8, zorder=0)
    ax.axhline(0, color="#999", lw=0.8, zorder=0)
    ax.set_xlim(0, lim)
    ax.set_ylim(-3.5, 6)
    ax.set_xlabel("|Explore z|  (shaded: just past significance, 1.96–3)")
    ax.set_ylabel("Validation z, in the explore direction")
    ax.set_title(
        f"{len(t)} ratio-effect hypotheses rerun on held-out patients", fontsize=10
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    sig = t[t["explore_p"] < 0.05].copy()
    ratio = sig["se_explore"] / sig["se_val"]
    sig["expected"] = norm.sf(1.96 - sig["z_explore"] * ratio)
    sig["observed"] = (sig["z_val"] > 0) & (sig["val_p"] < 0.05)
    groups = [("all", sig)] + [(g, sig[sig["group"] == g]) for g in ("few", "many")]
    x = np.arange(len(groups))
    exp = [s["expected"].sum() for _, s in groups]
    obs = [s["observed"].sum() for _, s in groups]
    w = 0.38
    bx.bar(x - w / 2, exp, w, color="#bbb", label="expected if explore were true")
    bx.bar(
        x + w / 2,
        obs,
        w,
        color=[colors.get(g, "#333") for g, _ in groups],
        label="observed",
    )
    for i, (_, s) in enumerate(groups):
        bx.text(i, max(exp[i], obs[i]) + 0.4, f"n={len(s)}", ha="center", fontsize=8.5)
    bx.set_xticks(
        x,
        [
            "all",
            labels["few"].replace(" before", "\nbefore"),
            labels["many"].replace(" before", "\nbefore"),
        ],
        fontsize=8.5,
    )
    bx.set_ylabel("Hypotheses replicated (val p<0.05, same sign)")
    bx.set_title("Explore-significant hypotheses: replications", fontsize=10)
    bx.legend(frameon=False, fontsize=8, loc="upper right")

    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print(sig.groupby("group")[["expected", "observed"]].sum())


if __name__ == "__main__":
    main()
