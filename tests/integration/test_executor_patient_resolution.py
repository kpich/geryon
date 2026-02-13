"""Integration test: PATIENT_ID resolution through JOIN for mutations_extended."""

from pathlib import Path

import pandas as pd

from msk_cycl.db import Database
from msk_cycl.engine import HypothesisExecutor
from msk_cycl.lang import CohortFilter, SelectCohort


def _write_parquet_tables(tmp_path: Path) -> None:
    """Create mutations_extended, clinical_sample, and clinical_patient parquets."""
    mutations = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S001", "S002", "S003", "S004"],
            "Hugo_Symbol": ["TP53", "KRAS", "TP53", "BRCA1"],
            "Variant_Classification": [
                "Missense_Mutation",
                "Missense_Mutation",
                "Nonsense_Mutation",
                "Missense_Mutation",
            ],
        }
    )
    mutations.to_parquet(tmp_path / "data_mutations_extended.parquet", index=False)

    clinical_sample = pd.DataFrame(
        {
            "SAMPLE_ID": ["S001", "S002", "S003", "S004"],
            "PATIENT_ID": ["P001", "P002", "P003", "P004"],
        }
    )
    clinical_sample.to_parquet(tmp_path / "data_clinical_sample.parquet", index=False)

    clinical_patient = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003", "P004"],
            "OS_MONTHS": [12.5, 8.3, 15.2, 9.1],
            "OS_STATUS": [1, 1, 0, 1],
        }
    )
    clinical_patient.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)


def test_get_cohort_ids_resolves_patient_ids_via_join(tmp_path: Path):
    """Executor resolves PATIENT_ID through clinical_sample JOIN for mutations."""
    _write_parquet_tables(tmp_path)

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)

        cohort = SelectCohort(
            filters=[
                CohortFilter(
                    table="mutations_extended",
                    column="Hugo_Symbol",
                    operator="==",
                    value="TP53",
                )
            ]
        )

        ids = executor._get_cohort_ids(cohort)

        assert sorted(ids) == ["P001", "P003"]


def test_get_cohort_ids_direct_path_still_works(tmp_path: Path):
    """Executor still uses direct SELECT for tables with PATIENT_ID."""
    _write_parquet_tables(tmp_path)

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)

        cohort = SelectCohort(
            filters=[
                CohortFilter(
                    table="clinical_patient",
                    column="OS_STATUS",
                    operator="==",
                    value=1,
                )
            ]
        )

        ids = executor._get_cohort_ids(cohort)

        assert sorted(ids) == ["P001", "P002", "P004"]
