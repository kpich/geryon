"""Tests for hypothesis spec compiler."""

import pytest

from msk_cycl.hypothesis.compiler import (
    compile_cohort_filter,
    compile_hypothesis,
    compile_select_cohort,
)
from msk_cycl.hypothesis.spec import CohortFilter, CyclHyp, SelectCohort


def test_compile_filter_equality_string() -> None:
    """Verify equality operator with string value."""
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="==",
        value="Lung Adenocarcinoma",
    )
    sql = compile_cohort_filter(filter)
    assert sql == "CANCER_TYPE = 'Lung Adenocarcinoma'"


def test_compile_filter_equality_numeric() -> None:
    """Verify equality operator with numeric value."""
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator="==", value=65
    )
    sql = compile_cohort_filter(filter)
    assert sql == "AGE = 65"


def test_compile_filter_not_equal() -> None:
    """Verify not-equal operator."""
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="!=",
        value="Unknown",
    )
    sql = compile_cohort_filter(filter)
    assert sql == "CANCER_TYPE != 'Unknown'"


def test_compile_filter_greater_than() -> None:
    """Verify greater-than operator."""
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator=">", value=65
    )
    sql = compile_cohort_filter(filter)
    assert sql == "AGE > 65"


def test_compile_filter_less_than_or_equal() -> None:
    """Verify less-than-or-equal operator."""
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator="<=", value=50
    )
    sql = compile_cohort_filter(filter)
    assert sql == "AGE <= 50"


def test_compile_filter_in_operator() -> None:
    """Verify IN operator with list of values."""
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="in",
        value=["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"],
    )
    sql = compile_cohort_filter(filter)
    assert (
        sql == "CANCER_TYPE IN ('Lung Adenocarcinoma', 'Lung Squamous Cell Carcinoma')"
    )


def test_compile_filter_in_operator_numeric() -> None:
    """Verify IN operator with numeric values."""
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator="in", value=[65, 70, 75]
    )
    sql = compile_cohort_filter(filter)
    assert sql == "AGE IN (65, 70, 75)"


def test_compile_filter_sql_injection_protection() -> None:
    """Verify SQL injection attempts are escaped."""
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="==",
        value="'; DROP TABLE patients; --",
    )
    sql = compile_cohort_filter(filter)
    # Single quotes should be escaped by doubling
    assert "''" in sql
    assert "DROP TABLE" in sql  # Still in the string, but escaped


def test_compile_filter_in_requires_list() -> None:
    """Verify IN operator requires list value."""
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="in",
        value="Lung Adenocarcinoma",  # String, not list
    )
    with pytest.raises(ValueError, match="'in' operator requires list value"):
        compile_cohort_filter(filter)


def test_compile_filter_in_requires_non_empty_list() -> None:
    """Verify IN operator requires non-empty list."""
    filter = CohortFilter(
        table="clinical_patient", column="CANCER_TYPE", operator="in", value=[]
    )
    with pytest.raises(ValueError, match="non-empty list"):
        compile_cohort_filter(filter)


def test_compile_select_cohort_single_filter() -> None:
    """Verify compilation of SelectCohort with single filter."""
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
        "SELECT DISTINCT PATIENT_ID FROM clinical_patient "
        "WHERE CANCER_TYPE = 'Lung Adenocarcinoma'"
    )


def test_compile_select_cohort_multiple_filters() -> None:
    """Verify compilation of SelectCohort with multiple filters (AND logic)."""
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
        "SELECT DISTINCT PATIENT_ID FROM clinical_patient "
        "WHERE CANCER_TYPE = 'Lung Adenocarcinoma' AND AGE > 65"
    )


def test_compile_select_cohort_custom_patient_id() -> None:
    """Verify compilation uses custom patient ID column."""
    query = SelectCohort(
        filters=[
            CohortFilter(
                table="clinical_patient",
                column="CANCER_TYPE",
                operator="==",
                value="Lung Adenocarcinoma",
            )
        ],
        patient_id_column="PATIENT_ID_CUSTOM",
    )
    sql = compile_select_cohort(query)
    assert "PATIENT_ID_CUSTOM" in sql


def test_compile_hypothesis_end_to_end() -> None:
    """Verify complete CyclHyp compilation."""
    spec = CyclHyp(
        query=SelectCohort(
            filters=[
                CohortFilter(
                    table="clinical_patient",
                    column="CANCER_TYPE",
                    operator="==",
                    value="Lung Adenocarcinoma",
                )
            ]
        )
    )
    sql = compile_hypothesis(spec)
    assert "SELECT DISTINCT PATIENT_ID" in sql
    assert "FROM clinical_patient" in sql
    assert "WHERE CANCER_TYPE = 'Lung Adenocarcinoma'" in sql
