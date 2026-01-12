"""Overall survival outcome handler."""

import pandas as pd  # type: ignore

from msk_cycl.db import Database
from msk_cycl.lang.outcomes import OverallSurvival


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
            DataFrame with columns: PATIENT_ID, time, event
        """
        ids_str = ", ".join(f"'{id}'" for id in cohort_ids)
        # Quote identifiers to handle special characters
        sql = f"""
            SELECT "PATIENT_ID", "{outcome.time_column}" as time,
                   "{outcome.event_column}" as event
            FROM "{outcome.table}"
            WHERE "PATIENT_ID" IN ({ids_str})
        """
        return db.execute(sql)
