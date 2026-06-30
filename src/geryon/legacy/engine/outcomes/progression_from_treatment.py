"""Progression-free survival from treatment outcome handler."""

import pandas as pd  # type: ignore

from geryon.db import Database
from geryon.legacy.lang.outcomes import ProgressionFromTreatment

DAYS_PER_MONTH = 30.44


class ProgressionFromTreatmentHandler:
    """Handler for extracting PFS-from-treatment outcome data.

    PFS = time from first dose of an agent to radiological progression or death,
    whichever comes first. Right-censored for patients alive without progression.
    """

    def extract_data(
        self,
        cohort_ids: list[str],
        outcome: ProgressionFromTreatment,
        db: Database,
    ) -> pd.DataFrame:
        """Extract PFS data anchored to first treatment with an agent.

        Logic:
        1. tx_start = first START_DATE for the agent
        2. first_prog = first START_DATE in timeline_progression where
           PROGRESSION='Yes' AND START_DATE > tx_start
        3. os_time = OS_MONTHS - tx_start / 30.44
        4. os_event = parsed OS_STATUS (1=deceased, 0=censored)
        5. If first_prog exists and prog_time < os_time: time=prog_time, event=1
        6. Else: time=os_time, event=os_event
        7. entry_time = max(0, -tx_start / 30.44): left-truncate treatment that
           preceded sequencing (the treatment->sequencing gap is immortal time)
        8. Exclude observed follow-up (time - entry_time) <= 0

        Returns DataFrame with columns: PATIENT_ID, time, event, entry_time
        """
        if not cohort_ids:
            return pd.DataFrame(columns=["PATIENT_ID", "time", "event", "entry_time"])

        ids_str = ", ".join(f"'{pid}'" for pid in cohort_ids)

        sql = f"""
            WITH first_tx AS (
                SELECT PATIENT_ID, MIN(START_DATE) AS tx_start
                FROM "{outcome.treatment_table}"
                WHERE PATIENT_ID IN ({ids_str})
                  AND AGENT = '{outcome.agent}'
                GROUP BY PATIENT_ID
            ),
            first_prog AS (
                SELECT p.PATIENT_ID, MIN(p.START_DATE) AS prog_date
                FROM "{outcome.progression_table}" p
                JOIN first_tx f ON p.PATIENT_ID = f.PATIENT_ID
                WHERE p.PROGRESSION = 'Yes'
                  AND p.START_DATE > f.tx_start
                GROUP BY p.PATIENT_ID
            )
            SELECT f.PATIENT_ID,
                   f.tx_start,
                   c.OS_MONTHS,
                   c.OS_STATUS,
                   fp.prog_date
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

        df["os_event"] = df["OS_STATUS"].astype(str).str.extract(r"^(\d+)").squeeze()
        df["os_event"] = pd.to_numeric(df["os_event"], errors="coerce")

        df["prog_time"] = (df["prog_date"] - df["tx_start"]) / DAYS_PER_MONTH

        prog_wins = df["prog_time"].notna() & (df["prog_time"] < df["os_time"])
        df["time"] = df["os_time"].copy()
        df["event"] = df["os_event"].copy()
        df.loc[prog_wins, "time"] = df.loc[prog_wins, "prog_time"]
        df.loc[prog_wins, "event"] = 1

        # Left-truncation entry: treatment before sequencing (tx_start < 0) enters
        # the risk set at sequencing; treatment at/after sequencing enters at 0.
        df["entry_time"] = (-df["tx_start"] / DAYS_PER_MONTH).clip(lower=0.0)

        df = df.dropna(subset=["time", "event"])
        # observed follow-up (time - entry_time) must be positive
        df = df[df["time"] - df["entry_time"] > 0]

        return df[["PATIENT_ID", "time", "event", "entry_time"]].reset_index(drop=True)
