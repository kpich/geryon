"""Tests for the replication predictors."""

import numpy as np
import pytest

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


def test_loo_logistic_ignores_a_constant_column():
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.normal(size=30), np.full(30, 2.0)])
    y = x[:, 0] > 0
    assert np.isfinite(loo_logistic(x, y)).all()


def test_loo_logistic_rejects_a_singleton_class():
    x = np.arange(4.0)[:, None]
    y = np.array([True, True, False, True])
    with pytest.raises(ValueError):
        loo_logistic(x, y)
