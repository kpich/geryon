"""Does the AI know which of its findings are overfit?

Every hypothesis is rerun on held-out patients. Power alone predicts a validation z
of |z_explore| · se_explore/se_val if the explore estimate were true. The residual,
observed minus that, is how much worse (or better) the finding did than its own
statistics promised: the part a smart reviewer would have to *know* something to
predict.

Left: the critic's forecast shrinkage (log predicted / log explore effect) against
|z_explore|, colored by how many scripts the generator ran before submitting. The
critic is told to account for selection inflation but never sees that search.
Right: Spearman ρ of each signal with the residual, with 90% bootstrap intervals.
Signals the AI produced are on top; the generator's search effort, which the critic
cannot see, is at the bottom. Ratio effects only.

    uv run python analysis/self_awareness.py   # -> plots/analysis/self_awareness.pdf
"""

import json
from pathlib import Path
import re

import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

OUT = (
    Path(__file__).resolve().parent.parent / "plots" / "analysis" / "self_awareness.pdf"
)
FEW = "#0072B2"
MANY = "#D55E00"
AI = "#555555"
HIDDEN = "#D55E00"
SPLIT = 12

HEDGE = "|".join(
    [
        "suggestive", "borderline", "weak", "modest", "trend", "nominal",
        "not significant", "underpowered", "exploratory", r"post[- ]hoc",
        "multiple (?:testing|comparisons)", "caution",
    ]
)  # fmt: skip
SELECTION = "|".join(
    [
        "multiple (?:testing|comparison)", "forking", "garden", r"data[- ]dredg",
        r"post[- ]hoc", "specification search", "cherry", "p-hack", "selected",
    ]
)  # fmt: skip


def _raw_hypotheses() -> dict[str, dict]:
    out = {}
    for path in load.SESSIONS.glob("*/*/hypotheses.jsonl"):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["record_type"] == "hypothesis":
                out[r["data"]["hypothesis_id"]] = r["data"]
    return out


def frame() -> pd.DataFrame:
    t = load.holdout()
    t = t[t["pred"].notna() & (t["pred"] > 0)].copy()
    sign = np.sign(t["z_explore"])
    t["abs_z"] = t["z_explore"].abs()
    t["z_val_dir"] = t["z_val"] * sign
    t["critic_shrink"] = np.log(t["pred"]) / t["log_explore"]
    t["resid"] = t["z_val_dir"] - t["abs_z"] * t["se_explore"] / t["se_val"]
    x = sm.add_constant(np.log(t["abs_z"]))
    t["critic_shrink_beyond_z"] = sm.OLS(t["critic_shrink"], x).fit().resid

    raw = _raw_hypotheses()

    def gen_text(i: str) -> str:
        d = raw[i]
        return " ".join([d["title"], d["description"], d.get("rationale") or ""])

    t["gen_hedges"] = t["hypothesis_id"].map(
        lambda i: len(re.findall(HEDGE, gen_text(i), re.I))
    )
    t["critic_flags_selection"] = t["hypothesis_id"].map(
        lambda i: bool(
            re.search(
                SELECTION, (raw[i].get("critique") or {}).get("notes") or "", re.I
            )
        )
    )
    return t


SIGNALS = [
    ("critic_shrink_beyond_z", "Critic forecast shrinkage, beyond |z|"),
    ("trust", "Critic trustworthiness"),
    ("holds_up", "Critic: holds up after its controls"),
    ("confound", "Critic confound risk (sign flipped)"),
    ("n_tests", "Critic: number of controls run"),
    ("critic_flags_selection", "Critic notes mention multiple testing / selection"),
    ("gen_hedges", "Generator hedging words in its write-up"),
    ("novelty", "Critic novelty (sign flipped)"),
    ("run_python", "Generator scripts before submit (sign flipped)"),
]
FLIP = {"confound", "novelty", "run_python"}


def _rho(x: pd.Series, y: pd.Series) -> float:
    return float(spearmanr(x.astype(float), y).statistic)


def main() -> None:
    t = frame()
    rng = np.random.default_rng(0)
    rows = []
    for col, label in SIGNALS:
        s = t[[col, "resid"]].dropna()
        sgn = -1 if col in FLIP else 1
        est = sgn * _rho(s[col], s["resid"])
        boots = []
        for _ in range(2000):
            b = s.iloc[rng.integers(0, len(s), len(s))]
            if b[col].nunique() > 1:
                boots.append(sgn * _rho(b[col], b["resid"]))
        lo, hi = np.nanpercentile(boots, [5, 95])
        rows.append((label, est, lo, hi, len(s), col == "run_python"))

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.35]}
    )

    grp = np.where(
        t["run_python"].isna(),
        "unknown",
        np.where(t["run_python"] <= SPLIT, "few", "many"),
    )
    for g, color, label in [
        ("unknown", "#999999", "not logged"),
        ("few", FEW, f"≤{SPLIT} generator scripts"),
        ("many", MANY, f">{SPLIT} generator scripts"),
    ]:
        s = t[grp == g]
        ax.scatter(
            s["abs_z"],
            s["critic_shrink"],
            s=36,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    k = float((t["se_explore"] / t["se_val"]).median())
    slope = float((t["abs_z"] * t["z_val_dir"]).sum() / (t["abs_z"] ** 2).sum())
    ax.axhline(
        slope / k, color="#333", lw=1.2, label=f"realized average ({slope / k:.2f})"
    )
    ax.axhline(1, color="#999", lw=0.8, ls="--", label="no shrinkage")
    ax.axvline(1.96, color="#ccc", lw=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 12)
    ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.set_ylim(-0.2, 1.4)
    ax.set_xlabel("|Explore z|")
    ax.set_ylabel("Critic forecast: log predicted / log explore effect")
    rho_z = _rho(t["abs_z"], t["critic_shrink"])
    has_n = t["run_python"].notna()
    rho_scripts = _rho(t.loc[has_n, "run_python"], t.loc[has_n, "critic_shrink"])
    ax.set_title(
        f"Critic shrinks weak results more (ρ={rho_z:.2f}),\n"
        f"barely tracks generator scripts (ρ={rho_scripts:.2f})",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    y = np.arange(len(rows))[::-1]
    for yi, (_, est, lo, hi, n, hidden) in zip(y, rows, strict=False):
        color = HIDDEN if hidden else AI
        bx.plot([lo, hi], [yi, yi], color=color, lw=2)
        bx.scatter([est], [yi], s=50, color=color, zorder=3)
        bx.text(
            1.02,
            yi,
            f"n={n}",
            va="center",
            fontsize=8,
            color="#666",
            transform=bx.get_yaxis_transform(),
        )
    bx.axvline(0, color="#999", lw=0.8)
    bx.set_yticks(y, [r[0] for r in rows], fontsize=8.5)
    bx.set_xlim(-0.75, 0.75)
    bx.set_xlabel(
        "Spearman ρ with replication beyond power\n"
        "(+ = signal says 'trust it' and it did better than its z promised)"
    )
    bx.set_title("Which signals know a finding will hold up?", fontsize=10)

    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    for r in rows:
        print(f"{r[0]:<55} {r[1]:+.2f}  [{r[2]:+.2f}, {r[3]:+.2f}]  n={r[4]}")


if __name__ == "__main__":
    main()
