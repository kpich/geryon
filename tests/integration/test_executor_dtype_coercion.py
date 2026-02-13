"""Integration test: executor succeeds with cBioPortal string-encoded OS_STATUS."""

from pathlib import Path

import pandas as pd
import pytest

from msk_cycl.db import Database
from msk_cycl.engine import HypothesisExecutor
from msk_cycl.lang import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    CyclHyp,
    OverallSurvival,
    SelectCohort,
)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::lifelines.utils.ConvergenceWarning")
def test_executor_succeeds_with_cbioportal_string_status(tmp_path: Path):
    """End-to-end: executor handles '1:DECEASED'/'0:LIVING' strings in OS_STATUS."""
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003", "P004", "P005", "P006"],
            "TREATMENT": [
                "Drug A",
                "Placebo",
                "Drug A",
                "Placebo",
                "Drug A",
                "Placebo",
            ],
            "OS_MONTHS": [12.5, 8.3, 15.2, 9.1, 20.0, 6.5],
            "OS_STATUS": [
                "1:DECEASED",
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
                "0:LIVING",
                "1:DECEASED",
            ],
        }
    )
    df.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)

        spec = CyclHyp(
            query=CompareCohorts(
                cohort_a=SelectCohort(
                    filters=[
                        CohortFilter(
                            table="clinical_patient",
                            column="TREATMENT",
                            operator="==",
                            value="Drug A",
                        )
                    ]
                ),
                cohort_b=SelectCohort(
                    filters=[
                        CohortFilter(
                            table="clinical_patient",
                            column="TREATMENT",
                            operator="==",
                            value="Placebo",
                        )
                    ]
                ),
                outcome=OverallSurvival(),
                method=ComparisonMethod.HAZARD_RATIO_COX,
            )
        )

        result = executor.execute(spec)

        assert result.success
        assert result.hazard_ratio is not None
        assert result.p_value is not None
        assert result.cohort_a_size == 3
        assert result.cohort_b_size == 3
