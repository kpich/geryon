"""Survival from treatment outcome handler."""

import pandas as pd  # type: ignore

from geryon.db import Database
from geryon.lang.outcomes import SurvivalFromTreatment

DAYS_PER_MONTH = 30.44


class SurvivalFromTreatmentHandler:
    """Handler for extracting survival-from-treatment outcome data."""

    def extract_data(
        self,
        cohort_ids: list[str],
        outcome: SurvivalFromTreatment,
        db: Database,
    ) -> pd.DataFrame:
        """Extract survival data anchored to first treatment with an agent.

        Logic:
        1. Find first START_DATE for the agent per patient
        2. Join with clinical_patient for OS_MONTHS and OS_STATUS
        3. time = OS_MONTHS - (START_DATE / 30.44)
        4. Parse cBioPortal OS_STATUS format
        5. Exclude patients with time <= 0 or without the treatment

        Returns DataFrame with columns: PATIENT_ID, time, event
        """
        if not cohort_ids:
            return pd.DataFrame(columns=["PATIENT_ID", "time", "event"])

        ids_str = ", ".join(f"'{pid}'" for pid in cohort_ids)

        sql = f"""
            WITH first_tx AS (
                SELECT PATIENT_ID, MIN(START_DATE) AS tx_start
                FROM "{outcome.table}"
                WHERE PATIENT_ID IN ({ids_str})
                  AND AGENT = '{outcome.agent}'
                GROUP BY PATIENT_ID
            )
            SELECT f.PATIENT_ID, f.tx_start,
                   c.OS_MONTHS, c.OS_STATUS
            FROM first_tx f
            JOIN clinical_patient c ON f.PATIENT_ID = c.PATIENT_ID
        """
        df = db.execute(sql)

        if df.empty:
            return pd.DataFrame(columns=["PATIENT_ID", "time", "event"])

        df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
        df["tx_start"] = pd.to_numeric(df["tx_start"], errors="coerce")

        df["time"] = df["OS_MONTHS"] - df["tx_start"] / DAYS_PER_MONTH

        # cBioPortal encodes OS_STATUS as "1:DECEASED"/"0:LIVING"
        df["event"] = df["OS_STATUS"].astype(str).str.extract(r"^(\d+)").squeeze()
        df["event"] = pd.to_numeric(df["event"], errors="coerce")

        df = df.dropna(subset=["time", "event"])
        df = df[df["time"] > 0]

        return df[["PATIENT_ID", "time", "event"]].reset_index(drop=True)
