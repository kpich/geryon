"""Tests for hypothesis spec compiler."""

import pytest

from msk_cycl.lang.compiler import (
    compile_cohort_filter,
    compile_select_cohort,
    compile_select_cohort_ids,
)
from msk_cycl.lang.spec import CohortFilter, SelectCohort


def test_equality_operator_with_string_compiles_correctly():
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="==",
        value="Lung Adenocarcinoma",
    )
    sql = compile_cohort_filter(filter)
    assert sql == "\"CANCER_TYPE\" = 'Lung Adenocarcinoma'"


def test_equality_operator_with_numeric_compiles_correctly():
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator="==", value=65
    )
    sql = compile_cohort_filter(filter)
    assert sql == '"AGE" = 65'


def test_not_equal_operator_compiles_correctly():
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="!=",
        value="Unknown",
    )
    sql = compile_cohort_filter(filter)
    assert sql == "\"CANCER_TYPE\" != 'Unknown'"


def test_greater_than_operator_compiles_correctly():
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator=">", value=65
    )
    sql = compile_cohort_filter(filter)
    assert sql == '"AGE" > 65'


def test_in_operator_with_string_list_compiles_correctly():
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="in",
        value=["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"],
    )
    sql = compile_cohort_filter(filter)
    assert sql == (
        "\"CANCER_TYPE\" IN ('Lung Adenocarcinoma', 'Lung Squamous Cell Carcinoma')"
    )


def test_in_operator_with_numeric_list_compiles_correctly():
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator="in", value=[65, 70, 75]
    )
    sql = compile_cohort_filter(filter)
    assert sql == '"AGE" IN (65, 70, 75)'


def test_sql_injection_attempt_is_escaped():
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="==",
        value="'; DROP TABLE patients; --",
    )
    sql = compile_cohort_filter(filter)
    assert "''" in sql


def test_in_operator_with_non_list_raises():
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="in",
        value="Lung Adenocarcinoma",
    )
    with pytest.raises(ValueError, match="'in' operator requires list value"):
        compile_cohort_filter(filter)


def test_in_operator_with_empty_list_raises():
    filter = CohortFilter(
        table="clinical_patient", column="CANCER_TYPE", operator="in", value=[]
    )
    with pytest.raises(ValueError, match="non-empty list"):
        compile_cohort_filter(filter)


def test_select_cohort_with_single_filter_compiles_correctly():
    query = SelectCohort(
        filters=[
            CohortFilter(
                table="clinical_patient",
                column="CANCER_TYPE",
                operator="==",
                value="Lung Adenocarcinoma",
            )
        ]
    )
    sql = compile_select_cohort(query)
    assert sql == (
        'SELECT * FROM "clinical_patient" '
        "WHERE \"CANCER_TYPE\" = 'Lung Adenocarcinoma'"
    )


def test_select_cohort_with_multiple_filters_uses_and_logic():
    query = SelectCohort(
        filters=[
            CohortFilter(
                table="clinical_patient",
                column="CANCER_TYPE",
                operator="==",
                value="Lung Adenocarcinoma",
            ),
            CohortFilter(
                table="clinical_patient", column="AGE", operator=">", value=65
            ),
        ]
    )
    sql = compile_select_cohort(query)
    assert sql == (
        'SELECT * FROM "clinical_patient" '
        'WHERE "CANCER_TYPE" = \'Lung Adenocarcinoma\' AND "AGE" > 65'
    )


def test_compile_select_cohort_ids_returns_only_patient_id():
    """compile_select_cohort_ids returns only PATIENT_ID column."""
    query = SelectCohort(
        filters=[
            CohortFilter(
                table="clinical_patient",
                column="CANCER_TYPE",
                operator="==",
                value="Lung Adenocarcinoma",
            )
        ]
    )
    sql = compile_select_cohort_ids(query)
    assert sql == (
        'SELECT "PATIENT_ID" FROM "clinical_patient" '
        "WHERE \"CANCER_TYPE\" = 'Lung Adenocarcinoma'"
    )
