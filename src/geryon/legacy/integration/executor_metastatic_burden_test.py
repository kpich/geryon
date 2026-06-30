"""Integration test: Metastatic burden outcome through HypothesisExecutor."""

from pathlib import Path

import pandas as pd

from geryon.db import Database
from geryon.legacy.engine import HypothesisExecutor
from geryon.legacy.lang import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    GeryonHyp,
    MetastaticBurden,
    SelectCohort,
)


def _write_fixtures(tmp_path: Path) -> None:
    clinical_patient = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2", "P3", "P4"],
            "CANCER_TYPE": ["Lung", "Lung", "Breast", "Breast"],
        }
    )
    clinical_patient.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)

    # P1: 3 sites near day 0, P2: 1 site
    # P3: 2 sites, P4: 0 sites (only "No Tumor Sites")
    timeline_tumor_sites = pd.DataFrame(
        {
            "PATIENT_ID": [
                "P1",
                "P1",
                "P1",
                "P2",
                "P3",
                "P3",
                "P4",
            ],
            "START_DATE": [
                -5,
                0,
                10,
                5,
                -10,
                15,
                0,
            ],
            "TUMOR_SITE": [
                "Lung",
                "Bone",
                "Liver",
                "Lymph Nodes",
                "Lung",
                "Bone",
                "No Tumor Sites",
            ],
        }
    )
    timeline_tumor_sites.to_parquet(
        tmp_path / "data_timeline_tumor_sites.parquet", index=False
    )


def test_metastatic_burden_end_to_end(tmp_path: Path):
    _write_fixtures(tmp_path)

    spec = GeryonHyp(
        query=CompareCohorts(
            cohort_a=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="CANCER_TYPE",
                        operator="==",
                        value="Lung",
                    )
                ]
            ),
            cohort_b=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="CANCER_TYPE",
                        operator="==",
                        value="Breast",
                    )
                ]
            ),
            outcome=MetastaticBurden(landmark_days=0, window_days=30),
            method=ComparisonMethod.WILCOXON_RANK_SUM,
        )
    )

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)
        result = executor.execute(spec)

    assert result.success
    assert result.cohort_a_size == 2
    assert result.cohort_b_size == 2
    # Lung (P1=3, P2=1) vs Breast (P3=2, P4=0)
    assert result.median_a is not None
    assert result.median_b is not None
    assert result.u_statistic is not None
    assert result.p_value is not None
