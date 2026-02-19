"""Tests for hypothesis spec Pydantic models."""

from pydantic import ValidationError
import pytest

from geryon.lang.methods import ComparisonMethod
from geryon.lang.outcomes import (
    MetastaticBurden,
    OverallSurvival,
    SurvivalFromTreatment,
    TimeToNextTreatment,
)
from geryon.lang.spec import (
    CohortFilter,
    CompareCohorts,
    GeryonHyp,
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
    """GeryonHyp defaults to version 1."""
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
    assert spec.version == 1


def test_cycl_hyp_with_invalid_version_raises():
    with pytest.raises(ValidationError):
        GeryonHyp(
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
    """GeryonHyp rejects bare SelectCohort (must use CompareCohorts)."""
    with pytest.raises(ValidationError):
        GeryonHyp(
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
    """GeryonHyp accepts CompareCohorts as query."""
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
    assert isinstance(spec.query, CompareCohorts)


# ---- New outcome type tests ----


def test_ttnt_defaults_and_required_agent():
    outcome = TimeToNextTreatment(agent="Pembrolizumab")
    assert outcome.outcome_type == "time_to_next_treatment"
    assert outcome.gap_days == 90
    assert outcome.table == "timeline_treatment"


def test_ttnt_requires_agent():
    with pytest.raises(ValidationError):
        TimeToNextTreatment()  # type: ignore


def test_survival_from_treatment_defaults():
    outcome = SurvivalFromTreatment(agent="Carboplatin")
    assert outcome.outcome_type == "survival_from_treatment"
    assert outcome.table == "timeline_treatment"


def test_metastatic_burden_defaults():
    outcome = MetastaticBurden()
    assert outcome.outcome_type == "metastatic_burden"
    assert outcome.landmark_days == 0
    assert outcome.window_days == 30
    assert outcome.table == "timeline_tumor_sites"


def test_discriminated_union_parses_os_from_dict():
    data = {
        "operation": "compare_cohorts",
        "cohort_a": {
            "operation": "select_cohort",
            "filters": [
                {
                    "table": "clinical_patient",
                    "column": "AGE",
                    "operator": ">",
                    "value": 50,
                }
            ],
        },
        "cohort_b": {
            "operation": "select_cohort",
            "filters": [
                {
                    "table": "clinical_patient",
                    "column": "AGE",
                    "operator": "<=",
                    "value": 50,
                }
            ],
        },
        "outcome": {"outcome_type": "overall_survival"},
        "method": "hazard_ratio_cox",
    }
    cc = CompareCohorts.model_validate(data)
    assert isinstance(cc.outcome, OverallSurvival)


def test_discriminated_union_parses_ttnt_from_dict():
    data = {
        "operation": "compare_cohorts",
        "cohort_a": {
            "operation": "select_cohort",
            "filters": [
                {
                    "table": "clinical_patient",
                    "column": "AGE",
                    "operator": ">",
                    "value": 50,
                }
            ],
        },
        "cohort_b": {
            "operation": "select_cohort",
            "filters": [
                {
                    "table": "clinical_patient",
                    "column": "AGE",
                    "operator": "<=",
                    "value": 50,
                }
            ],
        },
        "outcome": {"outcome_type": "time_to_next_treatment", "agent": "Pembro"},
        "method": "hazard_ratio_cox",
    }
    cc = CompareCohorts.model_validate(data)
    assert isinstance(cc.outcome, TimeToNextTreatment)
    assert cc.outcome.agent == "Pembro"


def test_discriminated_union_parses_metastatic_burden_from_dict():
    data = {
        "operation": "compare_cohorts",
        "cohort_a": {
            "operation": "select_cohort",
            "filters": [
                {
                    "table": "clinical_patient",
                    "column": "AGE",
                    "operator": ">",
                    "value": 50,
                }
            ],
        },
        "cohort_b": {
            "operation": "select_cohort",
            "filters": [
                {
                    "table": "clinical_patient",
                    "column": "AGE",
                    "operator": "<=",
                    "value": 50,
                }
            ],
        },
        "outcome": {"outcome_type": "metastatic_burden"},
        "method": "wilcoxon_rank_sum",
    }
    cc = CompareCohorts.model_validate(data)
    assert isinstance(cc.outcome, MetastaticBurden)


# ---- Method-outcome compatibility tests ----

_COHORT_A = SelectCohort(
    filters=[CohortFilter(table="t", column="c", operator="==", value=1)]
)
_COHORT_B = SelectCohort(
    filters=[CohortFilter(table="t", column="c", operator="==", value=0)]
)


def test_cox_with_ttnt_is_valid():
    CompareCohorts(
        cohort_a=_COHORT_A,
        cohort_b=_COHORT_B,
        outcome=TimeToNextTreatment(agent="X"),
        method=ComparisonMethod.HAZARD_RATIO_COX,
    )


def test_cox_with_survival_from_treatment_is_valid():
    CompareCohorts(
        cohort_a=_COHORT_A,
        cohort_b=_COHORT_B,
        outcome=SurvivalFromTreatment(agent="X"),
        method=ComparisonMethod.HAZARD_RATIO_COX,
    )


def test_cox_with_metastatic_burden_raises():
    with pytest.raises(ValidationError, match="time-to-event"):
        CompareCohorts(
            cohort_a=_COHORT_A,
            cohort_b=_COHORT_B,
            outcome=MetastaticBurden(),
            method=ComparisonMethod.HAZARD_RATIO_COX,
        )


def test_wilcoxon_with_metastatic_burden_is_valid():
    CompareCohorts(
        cohort_a=_COHORT_A,
        cohort_b=_COHORT_B,
        outcome=MetastaticBurden(),
        method=ComparisonMethod.WILCOXON_RANK_SUM,
    )


def test_wilcoxon_with_os_raises():
    with pytest.raises(ValidationError, match="continuous"):
        CompareCohorts(
            cohort_a=_COHORT_A,
            cohort_b=_COHORT_B,
            outcome=OverallSurvival(),
            method=ComparisonMethod.WILCOXON_RANK_SUM,
        )


def test_backward_compat_existing_os_json():
    """Existing OS JSON from prior sessions still parses correctly."""
    raw = {
        "version": 1,
        "query": {
            "operation": "compare_cohorts",
            "cohort_a": {
                "operation": "select_cohort",
                "filters": [
                    {
                        "table": "clinical_patient",
                        "column": "CANCER_TYPE",
                        "operator": "==",
                        "value": "Lung",
                    }
                ],
            },
            "cohort_b": {
                "operation": "select_cohort",
                "filters": [
                    {
                        "table": "clinical_patient",
                        "column": "CANCER_TYPE",
                        "operator": "==",
                        "value": "Breast",
                    }
                ],
            },
            "outcome": {
                "outcome_type": "overall_survival",
                "time_column": "OS_MONTHS",
                "event_column": "OS_STATUS",
                "table": "clinical_patient",
            },
            "method": "hazard_ratio_cox",
        },
    }
    spec = GeryonHyp.model_validate(raw)
    assert isinstance(spec.query.outcome, OverallSurvival)
    assert spec.query.outcome.time_column == "OS_MONTHS"
