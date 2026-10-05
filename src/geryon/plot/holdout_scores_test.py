"""Tests for the forecast scoring rule."""

import numpy as np

from geryon.plot.holdout_scores import interval_score


def test_interval_score_inside_is_width():
    assert interval_score(np.array([0.0]), np.array([2.0]), np.array([1.0]))[0] == 2.0


def test_interval_score_penalizes_misses_by_two_over_alpha():
    # 80% interval: alpha=0.2, a miss of 1 below adds 2/0.2 = 10.
    got = interval_score(np.array([0.0]), np.array([2.0]), np.array([-1.0]), level=0.8)
    assert np.isclose(got[0], 12.0)
