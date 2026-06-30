"""Metastatic burden outcome handler."""

import pandas as pd  # type: ignore

from geryon.db import Database
from geryon.legacy.lang.outcomes import MetastaticBurden


class MetastaticBurdenHandler:
    """Handler for extracting metastatic burden (distinct tumor site count)."""

    def extract_data(
        self,
        cohort_ids: list[str],
        outcome: MetastaticBurden,
        db: Database,
    ) -> pd.DataFrame:
        """Extract metastatic burden for a cohort.

        Logic:
        1. Query timeline_tumor_sites within [landmark - window, landmark + window]
        2. Count distinct TUMOR_SITE per patient, excluding "No Tumor Sites"
        3. Patients in cohort with no scans near landmark get value = 0

        Returns DataFrame with columns: PATIENT_ID, value
        """
        if not cohort_ids:
            return pd.DataFrame(columns=["PATIENT_ID", "value"])

        ids_str = ", ".join(f"'{pid}'" for pid in cohort_ids)
        lo = outcome.landmark_days - outcome.window_days
        hi = outcome.landmark_days + outcome.window_days

        sql = f"""
            SELECT PATIENT_ID, COUNT(DISTINCT TUMOR_SITE) AS value
            FROM "{outcome.table}"
            WHERE PATIENT_ID IN ({ids_str})
              AND START_DATE >= {lo}
              AND START_DATE <= {hi}
              AND TUMOR_SITE != 'No Tumor Sites'
            GROUP BY PATIENT_ID
        """
        df = db.execute(sql)

        # Patients in cohort with no matching scans get value = 0
        all_patients = pd.DataFrame({"PATIENT_ID": cohort_ids})
        result = all_patients.merge(df, on="PATIENT_ID", how="left")
        result["value"] = (
            pd.to_numeric(result["value"], errors="coerce").fillna(0).astype(int)
        )

        return result[["PATIENT_ID", "value"]]
