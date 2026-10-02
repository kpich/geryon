"""Tests for the holdout forecast baselines."""

import math

import pytest

from geryon.plot.holdout_table import explore_forecast, explore_se, is_ratio


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
