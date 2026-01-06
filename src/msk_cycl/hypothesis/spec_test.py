"""Tests for hypothesis spec Pydantic models."""

from pydantic import ValidationError
import pytest

from msk_cycl.hypothesis.spec import CohortFilter, CyclHyp, SelectCohort


def test_cohort_filter_with_string_value_is_valid():
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="==",
        value="Lung Adenocarcinoma",
    )
    assert filter.table == "clinical_patient"
    assert filter.value == "Lung Adenocarcinoma"


def test_cohort_filter_with_list_value_is_valid():
    filter = CohortFilter(
        table="clinical_patient",
        column="CANCER_TYPE",
        operator="in",
        value=["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"],
    )
    assert isinstance(filter.value, list)
    assert len(filter.value) == 2


def test_cohort_filter_with_invalid_operator_raises():
    with pytest.raises(ValidationError):
        CohortFilter(
            table="clinical_patient",
            column="CANCER_TYPE",
            operator="LIKE",  # type: ignore
            value="Lung%",
        )


def test_cycl_hyp_defaults_to_version_1():
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
    assert spec.version == 1


def test_cycl_hyp_with_invalid_version_raises():
    with pytest.raises(ValidationError):
        CyclHyp(
            version=99,  # type: ignore
            query=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="CANCER_TYPE",
                        operator="==",
                        value="Lung",
                    )
                ]
            ),
        )
