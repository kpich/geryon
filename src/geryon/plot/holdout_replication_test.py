"""Tests for the replication predictors."""

import numpy as np

from geryon.plot.holdout_replication import auc, loo_logistic


def test_auc_perfect_inverse_and_ties():
    label = np.array([True, True, False, False])
    assert auc(np.array([3, 4, 1, 2]), label) == 1.0
    assert auc(np.array([1, 2, 3, 4]), label) == 0.0
    assert auc(np.array([1, 1, 1, 1]), label) == 0.5


def test_loo_logistic_ranks_a_strong_predictor():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 2))
    y = x[:, 0] + 0.3 * rng.normal(size=60) > 0
    assert auc(loo_logistic(x, y), y) > 0.85
