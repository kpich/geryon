"""Integration test: multi-table cohort filter resolution."""

from pathlib import Path

import pandas as pd

from msk_cycl.db import Database
from msk_cycl.engine import HypothesisExecutor
from msk_cycl.lang import CohortFilter, SelectCohort


def _write_parquet_tables(tmp_path: Path) -> None:
    """Create clinical_patient, clinical_sample, and mutations_extended parquets."""
    clinical_patient = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003", "P004"],
            "CANCER_TYPE": ["Lung", "Lung", "Breast", "Lung"],
        }
    )
    clinical_patient.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)

    clinical_sample = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003", "P004"],
            "SAMPLE_ID": ["S001", "S002", "S003", "S004"],
        }
    )
    clinical_sample.to_parquet(tmp_path / "data_clinical_sample.parquet", index=False)

    mutations = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S001", "S002", "S003"],
            "Hugo_Symbol": ["KEAP1", "KEAP1", "TP53"],
        }
    )
    mutations.to_parquet(tmp_path / "data_mutations_extended.parquet", index=False)


def test_multi_table_cohort_intersects_patient_ids(tmp_path: Path):
    """Filters on clinical_patient + mutations_extended return correct intersection."""
    _write_parquet_tables(tmp_path)

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)
        cohort = SelectCohort(
            filters=[
                CohortFilter(
                    table="clinical_patient",
                    column="CANCER_TYPE",
                    operator="==",
                    value="Lung",
                ),
                CohortFilter(
                    table="mutations_extended",
                    column="Hugo_Symbol",
                    operator="==",
                    value="KEAP1",
                ),
            ]
        )

        ids = executor._get_cohort_ids(cohort)

        assert ids == ["P001", "P002"]
