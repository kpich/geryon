"""Base classes and protocols for outcome handlers."""

from typing import Protocol

import pandas as pd  # type: ignore

from geryon.db import Database


class OutcomeHandler(Protocol):
    """Protocol for handling outcome-specific data extraction.

    Each outcome type (OverallSurvival, etc.) has a corresponding handler
    that knows how to extract the relevant data from the database.
    """

    def extract_data(
        self, cohort_ids: list[str], outcome: object, db: Database
    ) -> pd.DataFrame:
        """Extract outcome data for a cohort.

        Parameters
        ----------
        cohort_ids : list[str]
            Patient/sample IDs in the cohort
        outcome : object
            Outcome specification (e.g., OverallSurvival instance)
        db : Database
            Database connection

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: PATIENT_ID, time, event
            (or other outcome-specific columns)
        """
        ...
