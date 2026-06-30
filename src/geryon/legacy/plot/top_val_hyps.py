"""Report top N hypotheses by val q-value (BH) as a text file."""

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from geryon.legacy.plot._loader import load_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="CSV from geryon.legacy.eval.batch"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Geryon data dir (contains sessions/)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["val_success"].astype(str) == "True"].copy()
    df = df[df["val_p_value"].notna()].copy()

    if df.empty:
        args.output.write_text("No successful val results.\n")
        return

    _, q_vals, _, _ = multipletests(
        df["val_p_value"].astype(float).to_numpy(), method="fdr_bh"
    )
    df["val_q_value"] = q_vals
    top = df.nsmallest(args.n, "val_q_value")

    hyp_by_id = {h.hypothesis_id: h for h in load_hypotheses(args.data_dir)}

    lines = [f"Top {args.n} hypotheses by val q-value (BH)\n"]
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        hid = row["hypothesis_id"]
        hyp = hyp_by_id.get(hid)
        lines.append(f"\n{'='*60}")
        lines.append(f"{rank}. {hid}")
        lines.append(
            f"   val q={row['val_q_value']:.4g}  val p={row['val_p_value']:.4g}  "
            f"train p={row['train_p_value']:.4g}"
        )
        if hyp is not None:
            p = hyp.proposal
            lines.append(f"   Cohort A:  {p.cohort_a_description}")
            lines.append(f"   Cohort B:  {p.cohort_b_description}")
            lines.append(f"   Outcome:   {p.outcome_description}")
            lines.append(f"   Rationale: {p.rationale}")
            if hyp.narrative is not None and hyp.narrative.summary:
                lines.append(f"   Summary:   {hyp.narrative.summary}")
        else:
            lines.append("   (hypothesis text not found in sessions)")

    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
