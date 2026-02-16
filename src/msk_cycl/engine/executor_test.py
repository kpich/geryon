"""Tests for hypothesis executor."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd  # type: ignore
import pytest

from msk_cycl.db import Database
from msk_cycl.engine import HypothesisExecutor
from msk_cycl.lang import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    ComparisonResult,
    CyclHyp,
    OverallSurvival,
    SelectCohort,
)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::lifelines.utils.ConvergenceWarning")
def test_executor_executes_compare_cohorts_query(tmp_path: Path):
    """Executor returns ComparisonResult for CompareCohorts queries."""
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003", "P004"],
            "TREATMENT": ["Drug A", "Placebo", "Drug A", "Placebo"],
            "OS_MONTHS": [12.5, 8.3, 15.2, 9.1],
            "OS_STATUS": [1, 1, 0, 1],
        }
    )
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

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

        assert isinstance(result, ComparisonResult)
        assert result.cohort_a_size == 2
        assert result.cohort_b_size == 2
        assert result.hazard_ratio is not None
        assert result.p_value is not None


def test_executor_get_cohort_ids_returns_list_of_patient_ids(tmp_path: Path):
    """_get_cohort_ids internal method returns list of patient IDs."""
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "TREATMENT": ["Drug A", "Placebo", "Drug A"],
        }
    )
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)

        cohort = SelectCohort(
            filters=[
                CohortFilter(
                    table="clinical_patient",
                    column="TREATMENT",
                    operator="==",
                    value="Drug A",
                )
            ]
        )

        ids = executor._get_cohort_ids(cohort)

        assert isinstance(ids, list)
        assert all(isinstance(id, str) for id in ids)
        assert ids == ["P001", "P003"]


def test_get_cohort_ids_dispatches_to_join_for_mutations_extended():
    """_get_cohort_ids uses JOIN path for tables in _PATIENT_ID_JOIN."""
    mock_db = MagicMock()
    mock_db.execute.return_value = pd.DataFrame({"PATIENT_ID": ["P001", "P002"]})

    executor = HypothesisExecutor(mock_db)
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

    assert ids == ["P001", "P002"]
    sql_called = mock_db.execute.call_args[0][0]
    assert "JOIN" in sql_called
    assert "clinical_sample" in sql_called


def test_get_cohort_ids_uses_direct_query_for_clinical_patient():
    """_get_cohort_ids uses direct SELECT for tables not in _PATIENT_ID_JOIN."""
    mock_db = MagicMock()
    mock_db.execute.return_value = pd.DataFrame({"PATIENT_ID": ["P001"]})

    executor = HypothesisExecutor(mock_db)
    cohort = SelectCohort(
        filters=[
            CohortFilter(
                table="clinical_patient",
                column="CANCER_TYPE",
                operator="==",
                value="Lung",
            )
        ]
    )

    ids = executor._get_cohort_ids(cohort)

    assert ids == ["P001"]
    sql_called = mock_db.execute.call_args[0][0]
    assert "JOIN" not in sql_called
    assert 'SELECT "PATIENT_ID" FROM "clinical_patient"' in sql_called
