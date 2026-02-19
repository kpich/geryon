"""Integration test: Survival from treatment outcome through HypothesisExecutor."""

from pathlib import Path

import pandas as pd

from geryon.db import Database
from geryon.engine import HypothesisExecutor
from geryon.lang import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    GeryonHyp,
    SelectCohort,
    SurvivalFromTreatment,
)


def _write_fixtures(tmp_path: Path) -> None:
    clinical_patient = pd.DataFrame(
        {
            "PATIENT_ID": [f"P{i}" for i in range(1, 11)],
            "CANCER_TYPE": ["Lung"] * 5 + ["Breast"] * 5,
            "OS_MONTHS": [36.0, 24.0, 48.0, 30.0, 42.0, 20.0, 15.0, 25.0, 35.0, 18.0],
            "OS_STATUS": [
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
                "1:DECEASED",
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
                "1:DECEASED",
            ],
        }
    )
    clinical_patient.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)

    # All patients received Carboplatin at various times
    timeline_treatment = pd.DataFrame(
        {
            "PATIENT_ID": [f"P{i}" for i in range(1, 11)],
            "START_DATE": [100, 200, 50, 300, 150, 80, 120, 90, 60, 200],
            "STOP_DATE": [110, 210, 60, 310, 160, 90, 130, 100, 70, 210],
            "AGENT": ["Carboplatin"] * 10,
            "SUBTYPE": ["Chemo"] * 10,
        }
    )
    timeline_treatment.to_parquet(
        tmp_path / "data_timeline_treatment.parquet", index=False
    )


def test_survival_from_treatment_end_to_end(tmp_path: Path):
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
            outcome=SurvivalFromTreatment(agent="Carboplatin"),
            method=ComparisonMethod.HAZARD_RATIO_COX,
        )
    )

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)
        result = executor.execute(spec)

    assert result.success
    assert result.cohort_a_size == 5
    assert result.cohort_b_size == 5
    assert result.hazard_ratio is not None
    assert result.p_value is not None
