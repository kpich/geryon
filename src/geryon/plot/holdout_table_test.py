"""Tests for the holdout forecast baselines."""

import math

import pytest

from geryon.codeflow.models import Expectation
from geryon.plot.holdout_table import (
    explore_forecast,
    explore_se,
    is_ratio,
    prediction_z,
)
from geryon.sandbox.result import IterationResult


@pytest.mark.parametrize(
    "text, expected",
    [
        ("adjusted HR (TTNT, delayed entry)", True),
        ("adjusted_HR_priorADCxindexADC_interaction", True),
        ("OR per +10% African ancestry", True),
        ("interaction HR ratio", True),
        ("hazard_ratio", True),
        ("adjusted difference in 6-month % BMI change (KEAP1-mut minus wt)", False),
        ("liver or bone", False),
        (None, False),
    ],
)
def test_is_ratio(text, expected):
    assert is_ratio(text) is expected


def test_se_from_ci_and_from_p_agree():
    se_ci = explore_se(2.0, 2.0 / math.exp(0.392), 2.0 * math.exp(0.392), None, True)
    assert se_ci == pytest.approx(0.2, rel=1e-3)
    p = 2 * (1 - 0.9999997133484281)  # z = 5 for log(2)/se below
    assert explore_se(2.0, None, None, p, True) == pytest.approx(
        math.log(2) / 5, rel=1e-3
    )


def test_explore_interval_is_widened_by_root_five_and_symmetric_on_log_scale():
    point, lo, hi = explore_forecast(
        2.0, 2.0 / math.exp(0.392), 2.0 * math.exp(0.392), None, True
    )
    assert point == 2.0
    assert lo is not None and hi is not None
    half = 1.2816 * 0.2 * math.sqrt(5)
    assert math.log(hi / point) == pytest.approx(half, rel=1e-3)
    assert math.log(point / lo) == pytest.approx(half, rel=1e-3)


def test_no_ci_or_p_gives_point_only():
    assert explore_forecast(-1.5, None, None, None, False) == (-1.5, None, None)


def _hr(effect: float, lo: float, hi: float) -> IterationResult:
    return IterationResult(
        effect_size=effect, effect_size_type="hazard_ratio", ci_lower=lo, ci_upper=hi
    )


def _expect(effect: float, lo: float, hi: float) -> Expectation:
    return Expectation(question="q", effect=effect, lower=lo, upper=hi)


def test_tight_null_against_an_expected_effect_scores_high():
    se = math.log(1.1 / 0.9) / (2 * 1.95996)
    spread = math.log(0.72 / 0.5) / (2 * 1.28155)
    got = prediction_z(_hr(1.0, 0.9, 1.1), _expect(0.6, 0.5, 0.72))
    assert got is not None
    assert got == pytest.approx(math.log(1 / 0.6) / math.hypot(se, spread), rel=1e-3)
    assert got > 3


def test_null_where_none_was_expected_scores_zero():
    assert prediction_z(_hr(1.0, 0.9, 1.1), _expect(1.0, 0.8, 1.25)) == 0


def test_a_vague_expectation_lowers_z():
    result = _hr(1.0, 0.9, 1.1)
    sharp = prediction_z(result, _expect(0.6, 0.5, 0.72))
    vague = prediction_z(result, _expect(0.6, 0.2, 1.8))
    assert sharp is not None and vague is not None and vague < sharp


def test_prediction_z_is_none_without_an_se_or_an_expectation():
    no_se = IterationResult(effect_size=1.0, effect_size_type="hazard_ratio")
    assert prediction_z(no_se, _expect(0.6, 0.5, 0.72)) is None
    assert prediction_z(_hr(1.0, 0.9, 1.1), None) is None
