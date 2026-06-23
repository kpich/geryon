"""Quantify how much left-truncation moves the stored treatment-hypothesis HRs.

Background: the OS handler models cohort entry at sequencing via `entry_time`
(left truncation), but `survival_from_treatment` / `progression_from_treatment`
never emit `entry_time`, so their stored HRs were fit *without* truncation. For
patients whose treatment started before sequencing (~52% of first doses), that
grants unmodeled immortal time and can bias the HR.

This script re-executes each stored treatment hypothesis two ways on the SAME
cohort — without entry_time (reproduces the stored fit) and with
entry_time = max(0, -tx_start/30.44) (corrected) — and reports how much the HR
moves. It does NOT modify the engine; it duplicates the handler SQL locally and
adds the tx_start needed to build entry_time.

Read-only against parquet + the public-free local DB. No LLM calls.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from pathlib import Path

import pandas as pd  # type: ignore

from geryon.db import Database
from geryon.engine import HypothesisExecutor, load_split_ids
from geryon.engine.methods import CoxHazardRatioMethod
from geryon.plot._loader import load_hypotheses
from geryon.tools.derived import replay_derived_views

DAYS_PER_MONTH = 30.44
TREATMENT_OUTCOMES = {"survival_from_treatment", "progression_from_treatment"}
HERE = os.path.dirname(os.path.abspath(__file__))


def _sql_str(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def _ids_in(ids: list[str]) -> str:
    return ", ".join(_sql_str(i) for i in ids)


def _add_entry_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """entry_time = months of immortal time before sequencing for pre-seq starts."""
    df["entry_time"] = (-df["tx_start"] / DAYS_PER_MONTH).clip(lower=0.0)
    # observed follow-up (time - entry_time) must be positive
    df = df[df["time"] - df["entry_time"] > 0].copy()
    return df[["PATIENT_ID", "time", "event", "entry_time"]]


def extract_survival_from_treatment(
    db: Database, ids: list[str], agent: str, table: str
) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=["PATIENT_ID", "time", "event", "entry_time"])
    sql = f"""
        WITH first_tx AS (
            SELECT PATIENT_ID, MIN(START_DATE) AS tx_start
            FROM "{table}"
            WHERE PATIENT_ID IN ({_ids_in(ids)}) AND AGENT = {_sql_str(agent)}
            GROUP BY PATIENT_ID
        )
        SELECT f.PATIENT_ID, f.tx_start, c.OS_MONTHS, c.OS_STATUS
        FROM first_tx f JOIN clinical_patient c ON f.PATIENT_ID = c.PATIENT_ID
    """
    df = db.execute(sql)
    if df.empty:
        return pd.DataFrame(columns=["PATIENT_ID", "time", "event", "entry_time"])
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["tx_start"] = pd.to_numeric(df["tx_start"], errors="coerce")
    df["time"] = df["OS_MONTHS"] - df["tx_start"] / DAYS_PER_MONTH
    df["event"] = pd.to_numeric(
        df["OS_STATUS"].astype(str).str.extract(r"^(\d+)").squeeze(), errors="coerce"
    )
    df = df.dropna(subset=["time", "event", "tx_start"])
    return _add_entry_and_filter(df)


def extract_progression_from_treatment(
    db: Database, ids: list[str], agent: str, tx_table: str, prog_table: str
) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(columns=["PATIENT_ID", "time", "event", "entry_time"])
    sql = f"""
        WITH first_tx AS (
            SELECT PATIENT_ID, MIN(START_DATE) AS tx_start
            FROM "{tx_table}"
            WHERE PATIENT_ID IN ({_ids_in(ids)}) AND AGENT = {_sql_str(agent)}
            GROUP BY PATIENT_ID
        ),
        first_prog AS (
            SELECT p.PATIENT_ID, MIN(p.START_DATE) AS prog_date
            FROM "{prog_table}" p JOIN first_tx f ON p.PATIENT_ID = f.PATIENT_ID
            WHERE p.PROGRESSION = 'Yes' AND p.START_DATE > f.tx_start
            GROUP BY p.PATIENT_ID
        )
        SELECT f.PATIENT_ID, f.tx_start, c.OS_MONTHS, c.OS_STATUS, fp.prog_date
        FROM first_tx f
        JOIN clinical_patient c ON f.PATIENT_ID = c.PATIENT_ID
        LEFT JOIN first_prog fp ON f.PATIENT_ID = fp.PATIENT_ID
    """
    df = db.execute(sql)
    if df.empty:
        return pd.DataFrame(columns=["PATIENT_ID", "time", "event", "entry_time"])
    df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    df["tx_start"] = pd.to_numeric(df["tx_start"], errors="coerce")
    df["prog_date"] = pd.to_numeric(df["prog_date"], errors="coerce")
    df["os_time"] = df["OS_MONTHS"] - df["tx_start"] / DAYS_PER_MONTH
    df["os_event"] = pd.to_numeric(
        df["OS_STATUS"].astype(str).str.extract(r"^(\d+)").squeeze(), errors="coerce"
    )
    df["prog_time"] = (df["prog_date"] - df["tx_start"]) / DAYS_PER_MONTH
    prog_wins = df["prog_time"].notna() & (df["prog_time"] < df["os_time"])
    df["time"] = df["os_time"]
    df["event"] = df["os_event"]
    df.loc[prog_wins, "time"] = df.loc[prog_wins, "prog_time"]
    df.loc[prog_wins, "event"] = 1
    df = df.dropna(subset=["time", "event", "tx_start"])
    return _add_entry_and_filter(df)


def _extract(db: Database, ids: list[str], outcome) -> pd.DataFrame:
    if outcome.outcome_type == "survival_from_treatment":
        return extract_survival_from_treatment(db, ids, outcome.agent, outcome.table)
    return extract_progression_from_treatment(
        db, ids, outcome.agent, outcome.treatment_table, outcome.progression_table
    )


def _cox(df_a: pd.DataFrame, df_b: pd.DataFrame, use_entry: bool):
    a = df_a if use_entry else df_a.drop(columns=["entry_time"], errors="ignore")
    b = df_b if use_entry else df_b.drop(columns=["entry_time"], errors="ignore")
    if len(a) < 2 or len(b) < 2 or a["event"].sum() < 1 or b["event"].sum() < 1:
        return None
    try:
        return CoxHazardRatioMethod().calculate(a, b)
    except Exception:
        return None


def _latest_parquet_dir() -> str:
    cands = sorted(glob.glob(os.path.expanduser("~/data/geryon_data/*")))
    cands = [c for c in cands if os.path.isdir(c)]
    if not cands:
        raise SystemExit("No parquet dir found under ~/data/geryon_data/")
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=os.path.expanduser("~/dev/cycl/geryon_data"))
    ap.add_argument(
        "--parquet-dir", default=None, help="default: latest ~/data/geryon_data/*"
    )
    ap.add_argument(
        "--split", default="train", help="'train' (matches original scoring) or 'all'"
    )
    ap.add_argument(
        "--output", default=os.path.join(HERE, "out", "truncation_audit.csv")
    )
    args = ap.parse_args()

    parquet_dir = args.parquet_dir or _latest_parquet_dir()
    print(f"parquet-dir: {parquet_dir}")
    data_dir = Path(args.data_dir)

    hyps = [
        h
        for h in load_hypotheses(data_dir)
        if h.spec.query.outcome.outcome_type in TREATMENT_OUTCOMES
    ]
    print(f"{len(hyps)} treatment hypotheses to audit")

    rows = []
    with Database(parquet_dir) as db:
        for views_path in sorted((data_dir / "sessions").rglob("derived_views.json")):
            replay_derived_views(db, views_path)

        pids = None if args.split == "all" else load_split_ids(db, args.split)
        executor = HypothesisExecutor(db, patient_ids=pids)

        for h in hyps:
            q = h.spec.query
            try:
                ids_a = executor._get_cohort_ids(q.cohort_a)
                ids_b = executor._get_cohort_ids(q.cohort_b)
            except Exception as e:
                rows.append(
                    {
                        "hypothesis_id": h.hypothesis_id,
                        "status": f"cohort_err:{type(e).__name__}",
                    }
                )
                continue

            df_a = _extract(db, ids_a, q.outcome)
            df_b = _extract(db, ids_b, q.outcome)
            pre_a = float((df_a["entry_time"] > 0).mean()) if len(df_a) else 0.0
            pre_b = float((df_b["entry_time"] > 0).mean()) if len(df_b) else 0.0

            untrunc = _cox(df_a, df_b, use_entry=False)
            trunc = _cox(df_a, df_b, use_entry=True)
            if untrunc is None or trunc is None:
                rows.append(
                    {
                        "hypothesis_id": h.hypothesis_id,
                        "status": "cox_failed",
                        "n_a": len(df_a),
                        "n_b": len(df_b),
                    }
                )
                continue

            hr_u, hr_t = untrunc["hazard_ratio"], trunc["hazard_ratio"]
            log2_shift = math.log2(hr_t / hr_u) if hr_u > 0 and hr_t > 0 else None
            crossed_one = (hr_u - 1) * (hr_t - 1) < 0  # HR moved to other side of 1
            p_u, p_t = untrunc["p_value"], trunc["p_value"]
            sig_changed = (p_u < 0.05) != (p_t < 0.05)
            rows.append(
                {
                    "hypothesis_id": h.hypothesis_id,
                    "outcome": q.outcome.outcome_type,
                    "agent": getattr(q.outcome, "agent", ""),
                    "stored_hr": h.result.hazard_ratio,
                    "stored_p": h.result.p_value,
                    "hr_untrunc": hr_u,
                    "p_untrunc": p_u,
                    "hr_trunc": hr_t,
                    "p_trunc": p_t,
                    "log2_hr_shift": log2_shift,
                    "hr_crossed_1": crossed_one,
                    "sig_changed_at_0.05": sig_changed,
                    "n_a": len(df_a),
                    "n_b": len(df_b),
                    "frac_preseq_a": round(pre_a, 3),
                    "frac_preseq_b": round(pre_b, 3),
                    "status": "ok",
                }
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fields = [
        "hypothesis_id",
        "outcome",
        "agent",
        "stored_hr",
        "stored_p",
        "hr_untrunc",
        "p_untrunc",
        "hr_trunc",
        "p_trunc",
        "log2_hr_shift",
        "hr_crossed_1",
        "sig_changed_at_0.05",
        "n_a",
        "n_b",
        "frac_preseq_a",
        "frac_preseq_b",
        "status",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"\nWrote {args.output}")
    print(f"  audited OK: {len(ok)}/{len(rows)}")
    if ok:
        repro = [abs(r["hr_untrunc"] - r["stored_hr"]) for r in ok if r["stored_hr"]]
        shifts = [abs(r["log2_hr_shift"]) for r in ok if r["log2_hr_shift"] is not None]
        print(
            "  reproduction check (|hr_untrunc - stored_hr|): "
            f"median={pd.Series(repro).median():.3f} max={max(repro):.3f}"
            if repro
            else "  (no stored HRs to compare)"
        )
        print(
            f"  |log2(HR_trunc/HR_untrunc)|: median={pd.Series(shifts).median():.3f} "
            f"max={max(shifts):.3f}  (0.585 = 1.5x, 1.0 = 2x)"
        )
        print(f"  HR crossed 1 (direction flip): {sum(r['hr_crossed_1'] for r in ok)}")
        print(
            f"  significance flipped at p<0.05: "
            f"{sum(r['sig_changed_at_0.05'] for r in ok)}"
        )


if __name__ == "__main__":
    main()
