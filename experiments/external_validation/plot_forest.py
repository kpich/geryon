"""Forest plot of independent TCGA overall-survival hazard ratios for the
genomic biomarkers underlying Geryon's MSK hypotheses.

Reads out/results.csv (from run.py), writes out/forest.png.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402  # type: ignore
import pandas as pd  # noqa: E402  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
SIG = "#c0392b"  # q < 0.05
NS = "#95a5a6"


def plot(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    df = df[df["hazard_ratio"].notna()].copy()
    # Clip absurd CIs (tiny arms) so the log axis stays readable.
    df["ci_lower"] = df["ci_lower"].clip(lower=0.05)
    df["ci_upper"] = df["ci_upper"].clip(upper=20)
    df = df.sort_values(["cancer_type", "hazard_ratio"], ascending=[False, True])
    df = df.reset_index(drop=True)

    y = np.arange(len(df))
    hr = df["hazard_ratio"].to_numpy()
    sig = (df["q_value"] < 0.05).fillna(False).to_numpy()
    colors = np.where(sig, SIG, NS)

    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(df) + 2.2))

    ax.hlines(
        y,
        df["ci_lower"].to_numpy(),
        df["ci_upper"].to_numpy(),
        color=colors,
        linewidth=1.4,
        alpha=0.8,
        zorder=2,
    )
    ax.scatter(hr, y, c=colors, s=34, zorder=3)
    ax.axvline(1.0, color="#34495e", lw=1, ls="--", zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(0.2, 6)
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_yticks(y)
    labels = [f"{r.label}  (n={int(r.n_altered)})" for r in df.itertuples()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.8, len(df) - 0.2)
    ax.set_xlabel(
        "TCGA overall-survival hazard ratio (altered vs wild-type), log scale"
    )
    ax.tick_params(axis="x", labelsize=9)

    ax.text(
        0.30,
        len(df) - 0.4,
        "← longer survival",
        fontsize=8,
        color="#34495e",
        ha="left",
        va="center",
    )
    ax.text(
        5.7,
        len(df) - 0.4,
        "shorter survival →",
        fontsize=8,
        color="#34495e",
        ha="right",
        va="center",
    )

    n_sig = int(sig.sum())
    ax.set_title(
        "Independent TCGA validation of Geryon hypothesis biomarkers\n"
        f"{len(df)} biomarker×cancer associations · {n_sig} significant at q<0.05 "
        "(BH) · red = significant",
        fontsize=11,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=SIG, label="q < 0.05"),
        plt.Line2D([], [], marker="o", ls="", color=NS, label="n.s."),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=os.path.join(HERE, "out", "results.csv"))
    ap.add_argument("--output", default=os.path.join(HERE, "out", "forest.png"))
    args = ap.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
