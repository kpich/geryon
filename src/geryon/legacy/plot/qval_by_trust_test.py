"""Tests for qval_by_trust data preparation (join + BH + −log10)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from geryon.legacy.plot.qval_by_trust import prepare_data


def _hyp(hid: str, trust: int | None):
    return SimpleNamespace(
        hypothesis_id=hid,
        effective_rating=SimpleNamespace(trustworthiness=trust),
    )


def test_prepare_joins_trust_and_computes_neg_log10_q():
    val_df = pd.DataFrame(
        {
            "hypothesis_id": ["a", "b", "c"],
            "val_p_value": [0.001, 0.04, 0.5],
            "val_success": [True, True, True],
        }
    )
    hyps = [_hyp("a", 3), _hyp("b", 2), _hyp("c", 1)]

    out = prepare_data(val_df, hyps).set_index("hypothesis_id")

    assert list(out["trust"]) == [3, 2, 1]
    _, expected_q, _, _ = multipletests([0.001, 0.04, 0.5], method="fdr_bh")
    np.testing.assert_allclose(out["val_q"].to_numpy(), expected_q)
    np.testing.assert_allclose(out["neg_log10_q"].to_numpy(), -np.log10(expected_q))


def test_failed_and_missing_val_get_nan_q_and_are_excluded_from_bh():
    val_df = pd.DataFrame(
        {
            "hypothesis_id": ["ok", "failed", "missing"],
            "val_p_value": [0.01, 0.01, np.nan],
            "val_success": [True, False, True],
        }
    )
    hyps = [_hyp("ok", 2), _hyp("failed", 2), _hyp("missing", 1)]

    out = prepare_data(val_df, hyps)  # row order preserved: ok, failed, missing
    val_q = out["val_q"].to_numpy()
    neg_log10_q = out["neg_log10_q"].to_numpy()

    # Only the single successful p-value is BH-corrected (q == p for n=1).
    assert np.isclose(val_q[0], 0.01)
    assert np.isnan(val_q[1])  # failed
    assert np.isnan(val_q[2])  # missing
    assert np.isnan(neg_log10_q[1])
    assert np.isnan(neg_log10_q[2])


def test_unrated_hypothesis_gets_nan_trust():
    val_df = pd.DataFrame(
        {"hypothesis_id": ["x"], "val_p_value": [0.02], "val_success": [True]}
    )
    out = prepare_data(val_df, [_hyp("x", None)])
    assert out["trust"].isna().all()
