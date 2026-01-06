"""Tests for hypothesis spec Pydantic models."""

from pydantic import ValidationError
import pytest

from msk_cycl.hypothesis.spec import CohortFilter, CyclHyp, SelectCohort


def test_cohort_filter_valid_string_value() -> None:
    """Verify CohortFilter accepts string values."""
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="==",
        value="Lung Adenocarcinoma",
    )
    assert filter.table == "clinical_patient"
    assert filter.value == "Lung Adenocarcinoma"


def test_cohort_filter_valid_numeric_value() -> None:
    """Verify CohortFilter accepts numeric values."""
    filter = CohortFilter(
        table="clinical_patient", column="AGE", operator=">", value=65
    )
    assert filter.value == 65


def test_cohort_filter_valid_list_value() -> None:
    """Verify CohortFilter accepts list values for 'in' operator."""
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="in",
        value=["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"],
    )
    assert isinstance(filter.value, list)
    assert len(filter.value) == 2


def test_cohort_filter_invalid_operator() -> None:
    """Verify CohortFilter rejects invalid operators."""
    with pytest.raises(ValidationError):
        CohortFilter(
            table="clinical_patient",
            column="CANCER_TYPE",
            operator="LIKE",  # type: ignore
            value="Lung%",
        )


def test_select_cohort_single_filter() -> None:
    """Verify SelectCohort works with single filter."""
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
    assert len(query.filters) == 1
    assert query.operation == "select_cohort"


def test_select_cohort_multiple_filters() -> None:
    """Verify SelectCohort works with multiple filters (AND logic)."""
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
    assert len(query.filters) == 2


def test_select_cohort_custom_patient_id_column() -> None:
    """Verify SelectCohort accepts custom patient ID column."""
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
    assert query.patient_id_column == "PATIENT_ID_CUSTOM"


def test_cycl_hyp_valid_spec() -> None:
    """Verify CyclHyp validates complete hypothesis spec."""
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
    assert isinstance(spec.query, SelectCohort)
    assert len(spec.query.filters) == 1
