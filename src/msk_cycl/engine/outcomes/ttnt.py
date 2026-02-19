"""Time to next treatment (TTNT) outcome handler."""

import pandas as pd  # type: ignore

from msk_cycl.db import Database
from msk_cycl.lang.outcomes import TimeToNextTreatment

DAYS_PER_MONTH = 30.44


class TimeToNextTreatmentHandler:
    """Handler for extracting TTNT outcome data."""

    def extract_data(
        self,
        cohort_ids: list[str],
        outcome: TimeToNextTreatment,
        db: Database,
    ) -> pd.DataFrame:
        """Extract TTNT data for a cohort.

        Logic:
        1. Find the index treatment: first START_DATE for the specified agent
        2. Find the next treatment: first record with
           START_DATE > index_START_DATE + gap_days (any agent)
        3. If next treatment found: time = (next - index) / 30.44, event = 1
        4. If no next treatment: censored with OS_MONTHS

        Returns DataFrame with columns: PATIENT_ID, time, event
        """
        if not cohort_ids:
            return pd.DataFrame(columns=["PATIENT_ID", "time", "event"])

        ids_str = ", ".join(f"'{pid}'" for pid in cohort_ids)

        sql = f"""
            WITH index_tx AS (
                SELECT PATIENT_ID, MIN(START_DATE) AS index_start
                FROM "{outcome.table}"
                WHERE PATIENT_ID IN ({ids_str})
                  AND AGENT = '{outcome.agent}'
                GROUP BY PATIENT_ID
            ),
            next_tx AS (
                SELECT t.PATIENT_ID, MIN(t.START_DATE) AS next_start
                FROM "{outcome.table}" t
                JOIN index_tx i ON t.PATIENT_ID = i.PATIENT_ID
                WHERE t.START_DATE > i.index_start + {outcome.gap_days}
                GROUP BY t.PATIENT_ID
            )
            SELECT i.PATIENT_ID, i.index_start, n.next_start
            FROM index_tx i
            LEFT JOIN next_tx n ON i.PATIENT_ID = n.PATIENT_ID
        """
        df = db.execute(sql)

        if df.empty:
            return pd.DataFrame(columns=["PATIENT_ID", "time", "event"])

        # Patients with a next treatment: event
        has_next = df["next_start"].notna()
        event_rows = df[has_next].copy()
        event_rows["time"] = (
            event_rows["next_start"] - event_rows["index_start"]
        ) / DAYS_PER_MONTH
        event_rows["event"] = 1.0

        # Patients without next treatment: censored using OS data
        censored_pids = df[~has_next]["PATIENT_ID"].tolist()
        if censored_pids:
            cens_ids_str = ", ".join(f"'{pid}'" for pid in censored_pids)
            os_sql = f"""
                SELECT PATIENT_ID, OS_MONTHS
                FROM clinical_patient
                WHERE PATIENT_ID IN ({cens_ids_str})
            """
            os_df = db.execute(os_sql)
            os_df["OS_MONTHS"] = pd.to_numeric(os_df["OS_MONTHS"], errors="coerce")

            cens = df[~has_next][["PATIENT_ID", "index_start"]].merge(
                os_df, on="PATIENT_ID", how="inner"
            )
            cens["time"] = cens["OS_MONTHS"] - cens["index_start"] / DAYS_PER_MONTH
            cens["event"] = 0.0
        else:
            cens = pd.DataFrame(
                {
                    "PATIENT_ID": pd.Series(dtype=str),
                    "time": pd.Series(dtype=float),
                    "event": pd.Series(dtype=float),
                }
            )

        parts = []
        if not event_rows.empty:
            parts.append(event_rows[["PATIENT_ID", "time", "event"]])
        if not cens.empty:
            parts.append(cens[["PATIENT_ID", "time", "event"]])

        if not parts:
            return pd.DataFrame(columns=["PATIENT_ID", "time", "event"])

        result = pd.concat(parts, ignore_index=True)

        result["time"] = pd.to_numeric(result["time"], errors="coerce")
        result["event"] = pd.to_numeric(result["event"], errors="coerce")
        result = result.dropna(subset=["time", "event"])
        result = result[result["time"] > 0]

        return result
