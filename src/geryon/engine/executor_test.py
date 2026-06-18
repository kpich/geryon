"""Tests for hypothesis executor."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd  # type: ignore
import pytest

from geryon.db import Database
from geryon.engine import HypothesisExecutor, load_split_ids
from geryon.lang import (
    CohortFilter,
    CompareCohorts,
    ComparisonMethod,
    ComparisonResult,
    GeryonHyp,
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

    dx_df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003", "P004"],
            "START_DATE": [-365.0, -365.0, -365.0, -365.0],
        }
    )
    (tmp_path / "data_timeline_diagnosis.parquet").write_bytes(
        dx_df.to_parquet(index=False)
    )

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)

        spec = GeryonHyp(
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


def test_get_cohort_ids_single_table_unchanged():
    """Single-table cohorts still take the direct fast path."""
    mock_db = MagicMock()
    mock_db.execute.return_value = pd.DataFrame({"PATIENT_ID": ["P001", "P003"]})

    executor = HypothesisExecutor(mock_db)
    cohort = SelectCohort(
        filters=[
            CohortFilter(
                table="clinical_patient",
                column="CANCER_TYPE",
                operator="==",
                value="Lung",
            ),
            CohortFilter(
                table="clinical_patient",
                column="SEX",
                operator="==",
                value="Female",
            ),
        ]
    )

    ids = executor._get_cohort_ids(cohort)

    assert ids == ["P001", "P003"]
    assert mock_db.execute.call_count == 1


def test_load_split_ids_returns_none_without_split_table():
    """load_split_ids returns None when patient_split table is absent."""
    mock_db = MagicMock()
    mock_db.list_tables.return_value = ["clinical_patient"]

    assert load_split_ids(mock_db, "train") is None


def test_load_split_ids_returns_frozenset_for_named_split(tmp_path: Path):
    """load_split_ids returns frozenset of IDs for the requested split name."""
    (tmp_path / "data_clinical_patient.parquet").write_bytes(
        pd.DataFrame({"PATIENT_ID": ["P001", "P002", "P003"]}).to_parquet(index=False)
    )
    split_df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "split": ["train", "validation", "train"],
        }
    )
    (tmp_path / "patient_split.parquet").write_bytes(split_df.to_parquet(index=False))

    with Database(tmp_path) as db:
        assert load_split_ids(db, "train") == frozenset(["P001", "P003"])
        assert load_split_ids(db, "validation") == frozenset(["P002"])


def test_get_cohort_ids_filters_to_supplied_patient_ids(tmp_path: Path):
    """_get_cohort_ids restricts results to patient_ids passed at construction."""
    clinical_df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003", "P004"],
            "TREATMENT": ["A", "A", "A", "A"],
        }
    )
    (tmp_path / "data_clinical_patient.parquet").write_bytes(
        clinical_df.to_parquet(index=False)
    )

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db, patient_ids=frozenset(["P001", "P003"]))
        cohort = SelectCohort(
            filters=[
                CohortFilter(
                    table="clinical_patient",
                    column="TREATMENT",
                    operator="==",
                    value="A",
                )
            ]
        )
        ids = executor._get_cohort_ids(cohort)

    assert ids == ["P001", "P003"]


def test_get_cohort_ids_no_filtering_without_split(tmp_path: Path):
    """_get_cohort_ids returns all matching patients when no split file exists."""
    clinical_df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "TREATMENT": ["A", "A", "B"],
            "OS_MONTHS": [10.0, 8.0, 12.0],
            "OS_STATUS": [1, 0, 1],
        }
    )
    (tmp_path / "data_clinical_patient.parquet").write_bytes(
        clinical_df.to_parquet(index=False)
    )

    with Database(tmp_path) as db:
        executor = HypothesisExecutor(db)
        cohort = SelectCohort(
            filters=[
                CohortFilter(
                    table="clinical_patient",
                    column="TREATMENT",
                    operator="==",
                    value="A",
                )
            ]
        )
        ids = executor._get_cohort_ids(cohort)

    assert ids == ["P001", "P002"]


def test_get_cohort_ids_multi_table_intersects():
    """Multi-table filters resolve each table independently and intersect."""
    mock_db = MagicMock()

    def fake_execute(sql: str) -> pd.DataFrame:
        if "clinical_patient" in sql:
            return pd.DataFrame({"PATIENT_ID": ["P001", "P002", "P003"]})
        if "mutations_extended" in sql:
            return pd.DataFrame({"PATIENT_ID": ["P002", "P003", "P004"]})
        raise ValueError(f"Unexpected SQL: {sql}")

    mock_db.execute.side_effect = fake_execute

    executor = HypothesisExecutor(mock_db)
    cohort = SelectCohort(
        filters=[
            CohortFilter(
                table="clinical_patient",
                column="CANCER_TYPE",
                operator="==",
                value="Non-Small Cell Lung Cancer",
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

    assert ids == ["P002", "P003"]
    assert mock_db.execute.call_count == 2

    calls = [call[0][0] for call in mock_db.execute.call_args_list]
    clinical_sql = [s for s in calls if "clinical_patient" in s][0]
    mutations_sql = [s for s in calls if "mutations_extended" in s][0]
    assert "JOIN" not in clinical_sql
    assert "JOIN" in mutations_sql
