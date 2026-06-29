"""One-off: does the KMT2B / MSS Pembrolizumab survival benefit survive a TMB control?

Hypothesis 1b996ae0-0b48-46a1-bb29-65b51795c458:
    Among microsatellite-stable (MSS), Pembrolizumab-treated patients, KMT2B-mutant
    tumors have better survival-from-treatment than KMT2B-wildtype (Cox).
    Reported: HR=0.68, train p=1.4e-6, val p=1.2e-3, val q(BH)=0.0183.

The natural next control: KMT2B-mutant tumors may simply carry higher TMB, and high
TMB drives immunotherapy benefit. MSS rules out the MSI-H hypermutator route, but NOT
the merely-high-TMB route. So we ask whether the KMT2B effect persists once TMB is
accounted for, three ways:

    (M0) baseline               event ~ kmt2b                 (reproduces the report)
    (M1) TMB-adjusted Cox       event ~ kmt2b + log1p(TMB)    (adjust TMB continuously)
    (M2) MSS TMB-LOW subgroup   event ~ kmt2b, TMB < 10       (within low-TMB tumors)
    (M3) MSS TMB-HIGH subgroup  event ~ kmt2b, TMB >= 10      (within high-TMB tumors)

TMB cutoff 10 mut/Mb = FDA pan-tumor pembrolizumab "TMB-high" threshold.

Run per split. NOTE: patient_split only contains `train` and `validation` -- there is
no separate `test` split, so the held-out set here is `validation`.

For validation we also recompute the batch BH q-value (from the eval val_results.csv),
substituting the TMB-adjusted p, to report whether the val q stays significant.

This is a throwaway analysis; it reuses geryon.db / the real survival-from-treatment
handler but is NOT wired into the main pipeline.

Usage:
    .venv/bin/python experiments/tmb_control/control_for_tmb.py \
        --parquet-dir ~/data/geryon_data/2026-06-18 \
        --val-csv <path/to/val_results.csv>
"""

import argparse
from pathlib import Path

from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from geryon.db import Database
from geryon.engine.outcomes.survival_from_treatment import SurvivalFromTreatmentHandler
from geryon.lang.outcomes import SurvivalFromTreatment

HYP_ID = "1b996ae0-0b48-46a1-bb29-65b51795c458"
GENE = "KMT2B"
AGENT = "Pembrolizumab"
TMB_HIGH_CUTOFF = 10.0  # mut/Mb; FDA pan-tumor pembrolizumab "TMB-high"


def build_frame(db: Database, split: str) -> pd.DataFrame:
    """Per-patient frame for the hypothesis's two cohorts in `split`.

    Cohorts reproduce the session's derived views exactly:
      A (KMT2B-mut MSS):  KMT2B mutation on a microsatellite-stable sample
      B (KMT2B-wt MSS):   stable on some sample, no KMT2B mutation on any sample
    TMB = max valid CVR_TMB_SCORE across the patient's samples.
    """
    split_ids = set(
        db.execute(f"SELECT PATIENT_ID FROM patient_split WHERE split = '{split}'")[
            "PATIENT_ID"
        ].tolist()
    )

    kmt2b_mss_ids = set(
        db.execute(
            f"""
            SELECT DISTINCT cs.PATIENT_ID
            FROM clinical_sample cs
            JOIN mutations_extended me ON cs.SAMPLE_ID = me.Tumor_Sample_Barcode
            WHERE me.Hugo_Symbol = '{GENE}' AND cs.MSI_TYPE = 'Stable'
            """
        )["PATIENT_ID"].tolist()
    )
    no_kmt2b_mss_ids = set(
        db.execute(
            f"""
            SELECT DISTINCT cs.PATIENT_ID
            FROM clinical_sample cs
            WHERE cs.MSI_TYPE = 'Stable' AND cs.PATIENT_ID NOT IN (
                SELECT DISTINCT cs2.PATIENT_ID
                FROM clinical_sample cs2
                JOIN mutations_extended me ON cs2.SAMPLE_ID = me.Tumor_Sample_Barcode
                WHERE me.Hugo_Symbol = '{GENE}'
            )
            """
        )["PATIENT_ID"].tolist()
    )
    kmt2b_mss_ids &= split_ids
    no_kmt2b_mss_ids &= split_ids
    cohort_ids = sorted(kmt2b_mss_ids | no_kmt2b_mss_ids)

    tmb = db.execute(
        """
        SELECT PATIENT_ID, MAX(CVR_TMB_SCORE) AS tmb
        FROM clinical_sample
        WHERE CVR_TMB_SCORE >= 0
        GROUP BY PATIENT_ID
        """
    )

    # Survival-from-treatment via the real handler; it inner-joins to Pembrolizumab,
    # so only MSS patients who actually received the agent come back.
    outcome = SurvivalFromTreatment(agent=AGENT)
    surv = SurvivalFromTreatmentHandler().extract_data(cohort_ids, outcome, db)

    surv["kmt2b"] = surv["PATIENT_ID"].isin(kmt2b_mss_ids).astype(int)
    surv = surv.merge(tmb, on="PATIENT_ID", how="left")
    surv["tmb_log"] = np.log1p(surv["tmb"])
    return surv


def fit_kmt2b(df: pd.DataFrame, covariates: list[str]) -> dict | None:
    """Cox fit (left-truncated); return the KMT2B row, or None if unfittable."""
    cols = ["time", "event", "entry_time"] + covariates
    data = df[cols].dropna()
    if data["event"].sum() < 5 or data["kmt2b"].nunique() < 2:
        return None
    cph = CoxPHFitter()
    cph.fit(data, duration_col="time", event_col="event", entry_col="entry_time")
    s = cph.summary.loc["kmt2b"]
    return {
        "n": len(data),
        "n_events": int(data["event"].sum()),
        "n_kmt2b": int(data["kmt2b"].sum()),
        "hr": float(cph.hazard_ratios_["kmt2b"]),
        "ci_low": float(np.exp(s["coef lower 95%"])),
        "ci_high": float(np.exp(s["coef upper 95%"])),
        "p": float(s["p"]),
    }


def fmt(r: dict | None) -> str:
    if r is None:
        return "  (not fittable: too few events or one-armed)"
    return (
        f"  HR={r['hr']:.3f} [{r['ci_low']:.3f}-{r['ci_high']:.3f}]  p={r['p']:.4g}"
        f"   n={r['n']} (events={r['n_events']}, KMT2B-mut={r['n_kmt2b']})"
    )


def recompute_val_q(val_csv: Path, new_p: float) -> tuple[float, float]:
    """Return (original val q, val q with this hyp's val_p replaced by new_p), BH."""
    df = pd.read_csv(val_csv)
    df = df[
        (df["val_success"].astype(str) == "True") & df["val_p_value"].notna()
    ].copy()

    orig = df["val_p_value"].astype(float).to_numpy()
    _, q_orig, _, _ = multipletests(orig, method="fdr_bh")
    idx = df.index[df["hypothesis_id"] == HYP_ID][0]
    pos = df.index.get_loc(idx)

    adj = orig.copy()
    adj[pos] = new_p
    _, q_adj, _, _ = multipletests(adj, method="fdr_bh")
    return float(q_orig[pos]), float(q_adj[pos])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True, type=Path)
    ap.add_argument(
        "--val-csv", required=True, type=Path, help="eval batch val_results.csv"
    )
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "out" / "tmb_control.txt"
    )
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    val_adjusted_p: dict[str, float] = {}

    with Database(args.parquet_dir) as db:
        for split in ("train", "validation"):
            df = build_frame(db, split)
            emit(f"\n{'='*72}\n{split.upper()}  (MSS + {AGENT}, n={len(df)})\n{'='*72}")

            mut = df[df["kmt2b"] == 1]
            wt = df[df["kmt2b"] == 0]
            with_tmb = df["tmb"].notna()
            emit(
                f"TMB available: {int(with_tmb.sum())}/{len(df)}.  "
                f"Median TMB  KMT2B-mut={mut['tmb'].median():.1f}  "
                f"wt={wt['tmb'].median():.1f}  "
                f"(KMT2B-mut TMB-high rate="
                f"{(mut['tmb'] >= TMB_HIGH_CUTOFF).mean():.2f} vs "
                f"wt={(wt['tmb'] >= TMB_HIGH_CUTOFF).mean():.2f})"
            )

            m0 = fit_kmt2b(df, ["kmt2b"])
            m1 = fit_kmt2b(df[with_tmb], ["kmt2b", "tmb_log"])
            low = df[df["tmb"] < TMB_HIGH_CUTOFF]
            high = df[df["tmb"] >= TMB_HIGH_CUTOFF]
            m2 = fit_kmt2b(low, ["kmt2b"])
            m3 = fit_kmt2b(high, ["kmt2b"])

            emit("M0 baseline      event ~ kmt2b")
            emit(fmt(m0))
            emit("M1 TMB-adjusted  event ~ kmt2b + log1p(TMB)")
            emit(fmt(m1))
            emit(f"M2 TMB-LOW (<{TMB_HIGH_CUTOFF:g})   event ~ kmt2b")
            emit(fmt(m2))
            emit(f"M3 TMB-HIGH (>={TMB_HIGH_CUTOFF:g})  event ~ kmt2b")
            emit(fmt(m3))

            if split == "validation":
                if m1 is not None:
                    val_adjusted_p["M1 TMB-adjusted"] = m1["p"]
                if m2 is not None:
                    val_adjusted_p["M2 TMB-low subgroup"] = m2["p"]

    emit(f"\n{'='*72}\nVALIDATION BH q-value (recomputed across eval batch)\n{'='*72}")
    for label, p in val_adjusted_p.items():
        q_orig, q_adj = recompute_val_q(args.val_csv, p)
        emit(
            f"{label}: val p={p:.4g}  ->  val q(BH)={q_adj:.4g}   "
            f"(baseline val q was {q_orig:.4g})"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    emit(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
