"""Integration test: TTNT outcome through HypothesisExecutor."""

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
    TimeToNextTreatment,
)


def _write_fixtures(tmp_path: Path) -> None:
    clinical_patient = pd.DataFrame(
        {
            "PATIENT_ID": [f"P{i}" for i in range(1, 11)],
            "CANCER_TYPE": ["Lung"] * 5 + ["Breast"] * 5,
            "OS_MONTHS": [36.0, 24.0, 48.0, 12.0, 30.0, 40.0, 20.0, 35.0, 25.0, 15.0],
            "OS_STATUS": [
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
                "1:DECEASED",
            ],
        }
    )
    clinical_patient.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)

    # All 10 patients have Pembrolizumab; some also have a subsequent treatment
    timeline_treatment = pd.DataFrame(
        {
            "PATIENT_ID": [
                "P1",
                "P1",
                "P2",
                "P3",
                "P3",
                "P4",
                "P5",
                "P6",
                "P6",
                "P7",
                "P8",
                "P8",
                "P9",
                "P10",
            ],
            "START_DATE": [
                100,
                300,
                200,
                50,
                250,
                80,
                150,
                60,
                400,
                120,
                90,
                350,
                70,
                110,
            ],
            "STOP_DATE": [
                110,
                310,
                210,
                60,
                260,
                90,
                160,
                70,
                410,
                130,
                100,
                360,
                80,
                120,
            ],
            "AGENT": [
                "Pembrolizumab",
                "Docetaxel",
                "Pembrolizumab",
                "Pembrolizumab",
                "Carboplatin",
                "Pembrolizumab",
                "Pembrolizumab",
                "Pembrolizumab",
                "Docetaxel",
                "Pembrolizumab",
                "Pembrolizumab",
                "Carboplatin",
                "Pembrolizumab",
                "Pembrolizumab",
            ],
            "SUBTYPE": [
                "Immuno",
                "Chemo",
                "Immuno",
                "Immuno",
                "Chemo",
                "Immuno",
                "Immuno",
                "Immuno",
                "Chemo",
                "Immuno",
                "Immuno",
                "Chemo",
                "Immuno",
                "Immuno",
            ],
        }
    )
    timeline_treatment.to_parquet(
        tmp_path / "data_timeline_treatment.parquet", index=False
    )


def test_ttnt_end_to_end(tmp_path: Path):
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
            outcome=TimeToNextTreatment(agent="Pembrolizumab", gap_days=90),
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
