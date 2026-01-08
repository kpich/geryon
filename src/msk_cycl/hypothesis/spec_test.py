"""Tests for hypothesis spec Pydantic models."""

from pydantic import ValidationError
import pytest

from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.spec import (
    CohortFilter,
    CompareCohorts,
    CyclHyp,
    SelectCohort,  # For negative tests only
)


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
    """CyclHyp defaults to version 1."""
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
    assert spec.version == 1


def test_cycl_hyp_with_invalid_version_raises():
    with pytest.raises(ValidationError):
        CyclHyp(
            version=99,  # type: ignore
            query=SelectCohort(  # type: ignore
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


def test_overall_survival_uses_cbioportal_defaults():
    """Overall survival defaults to standard cBioPortal column names."""
    outcome = OverallSurvival()
    assert outcome.time_column == "OS_MONTHS"
    assert outcome.event_column == "OS_STATUS"
    assert outcome.table == "clinical_patient"


def test_overall_survival_allows_custom_columns():
    """Overall survival supports custom column names for non-standard schemas."""
    outcome = OverallSurvival(
        time_column="SURVIVAL_TIME", event_column="EVENT", table="custom_clinical"
    )
    assert outcome.time_column == "SURVIVAL_TIME"
    assert outcome.event_column == "EVENT"
    assert outcome.table == "custom_clinical"


def test_compare_cohorts_with_valid_cox_and_os_succeeds():
    """Compare cohorts validates when Cox method is used with OS outcome."""
    comparison = CompareCohorts(
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
    assert comparison.method == ComparisonMethod.HAZARD_RATIO_COX


def test_cycl_hyp_rejects_select_cohort_query():
    """CyclHyp rejects bare SelectCohort (must use CompareCohorts)."""
    with pytest.raises(ValidationError):
        CyclHyp(
            query=SelectCohort(  # type: ignore
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="AGE",
                        operator=">",
                        value=65,
                    )
                ]
            )
        )


def test_cycl_hyp_accepts_compare_cohorts_query():
    """CyclHyp accepts CompareCohorts as query."""
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
    assert isinstance(spec.query, CompareCohorts)
