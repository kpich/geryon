"""Tests for hypothesis executor."""

from pathlib import Path

import pandas as pd  # type: ignore
import pytest

from msk_cycl.hypothesis.db import Database
from msk_cycl.hypothesis.executor import HypothesisExecutor
from msk_cycl.hypothesis.spec import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    CyclHyp,
    OverallSurvival,
    SelectCohort,
)


def test_executor_executes_select_cohort_query(tmp_path: Path):
    """Executor returns DataFrame for SelectCohort queries."""
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "TREATMENT": ["Drug A", "Placebo", "Drug A"],
            "AGE": [65, 70, 68],
        }
    )
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    db = Database(tmp_path)
    executor = HypothesisExecutor(db)

    spec = CyclHyp(
        query=SelectCohort(
            filters=[
                CohortFilter(
                    table="clinical_patient",
                    column="TREATMENT",
                    operator="==",
                    value="Drug A",
                )
            ]
        )
    )

    result = executor.execute(spec)

    assert result.query_type == "select_cohort"
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == 2
    assert result.data["PATIENT_ID"].tolist() == ["P001", "P003"]

    db.close()


def test_executor_raises_not_implemented_for_compare_cohorts(tmp_path: Path):
    """Executor raises NotImplementedError for CompareCohorts (stub)."""
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002"],
            "TREATMENT": ["Drug A", "Placebo"],
            "OS_MONTHS": [12.5, 8.3],
            "OS_STATUS": [1, 1],
        }
    )
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    db = Database(tmp_path)
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

    with pytest.raises(
        NotImplementedError, match="Cohort comparison execution not yet implemented"
    ):
        executor.execute(spec)

    db.close()


def test_executor_works_with_database_context_manager(tmp_path: Path):
    """Executor works with Database context manager pattern."""
    df = pd.DataFrame({"PATIENT_ID": ["P001"], "AGE": [65]})
    parquet_file = tmp_path / "data_clinical_patient.parquet"
    df.to_parquet(parquet_file, index=False)

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)

        spec = CyclHyp(
            query=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="AGE",
                        operator=">=",
                        value=65,
                    )
                ]
            )
        )

        result = executor.execute(spec)
        assert len(result.data) == 1
