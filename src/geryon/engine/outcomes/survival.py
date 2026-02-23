"""Overall survival outcome handler."""

import pandas as pd  # type: ignore

from geryon.db import Database
from geryon.lang.outcomes import OverallSurvival


def _process_os_df(df: pd.DataFrame) -> pd.DataFrame:
    # cBioPortal encodes OS_STATUS as "1:DECEASED"/"0:LIVING"
    # Extract leading digit so both formats parse to numeric
    df["event"] = df["event"].astype(str).str.extract(r"^(\d+)").squeeze()

    df["os_months"] = pd.to_numeric(df["os_months"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce")
    # .copy() ensures post-dropna assignments don't trigger SettingWithCopyWarning
    df = df.dropna(subset=["os_months", "event", "dx_days"]).copy()

    # dx_days < 0; subtraction adds the pre-sequencing period in months
    df["entry_time"] = -df["dx_days"] / 30.44
    df["time"] = df["os_months"] - df["dx_days"] / 30.44

    # Rows where time <= entry_time have zero or negative observed duration
    df = df[df["time"] > df["entry_time"]].copy()

    return df[["PATIENT_ID", "time", "event", "entry_time"]]


class OverallSurvivalHandler:
    """Handler for extracting overall survival outcome data."""

    def extract_data(
        self, cohort_ids: list[str], outcome: OverallSurvival, db: Database
    ) -> pd.DataFrame:
        """Extract overall survival data for a cohort.

        Parameters
        ----------
        cohort_ids : list[str]
            Patient/sample IDs in the cohort
        outcome : OverallSurvival
            Outcome specification with column names
        db : Database
            Database connection

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: PATIENT_ID, time, event, entry_time
        """
        ids_str = ", ".join(f"'{id}'" for id in cohort_ids)
        # START_DATEs in timeline_diagnosis are negative (days before sequencing=day 0).
        # MAX(START_DATE) where START_DATE < 0 picks the least-negative value = most
        # recent diagnosis before sequencing. Diagnoses on/after day 0 are excluded
        # since they aren't a meaningful prior anchor.
        # INNER JOIN drops patients with no prior diagnosis — no anchor means no
        # diagnosis-relative time axis, so they can't enter the analysis.
        sql = f"""
            WITH dx AS (
                SELECT "PATIENT_ID", MAX("START_DATE") AS dx_days
                FROM "timeline_diagnosis"
                WHERE "PATIENT_ID" IN ({ids_str})
                  AND "START_DATE" < 0
                GROUP BY "PATIENT_ID"
            )
            SELECT
                p."PATIENT_ID",
                p."{outcome.time_column}" AS os_months,
                p."{outcome.event_column}" AS event,
                d.dx_days
            FROM "{outcome.table}" p
            INNER JOIN dx d ON p."PATIENT_ID" = d."PATIENT_ID"
            WHERE p."PATIENT_ID" IN ({ids_str})
        """
        return _process_os_df(db.execute(sql))

    def load_all_data(self, outcome: OverallSurvival, db: Database) -> pd.DataFrame:
        """Load OS data for ALL patients in a single query."""
        sql = f"""
            WITH cp AS (
                SELECT "PATIENT_ID" FROM "{outcome.table}"
            ),
            dx AS (
                SELECT t."PATIENT_ID", MAX(t."START_DATE") AS dx_days
                FROM "timeline_diagnosis" t
                INNER JOIN cp ON t."PATIENT_ID" = cp."PATIENT_ID"
                WHERE t."START_DATE" < 0
                GROUP BY t."PATIENT_ID"
            )
            SELECT
                p."PATIENT_ID",
                p."{outcome.time_column}" AS os_months,
                p."{outcome.event_column}" AS event,
                d.dx_days
            FROM "{outcome.table}" p
            INNER JOIN dx d ON p."PATIENT_ID" = d."PATIENT_ID"
        """
        return _process_os_df(db.execute(sql))
