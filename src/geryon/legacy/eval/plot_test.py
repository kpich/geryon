"""Tests for eval plotting guards against degenerate inputs."""

import numpy as np
import pandas as pd

from geryon.legacy.eval.plot import plot_pval_comparison, plot_qval_comparison


def _df(train, val):
    return pd.DataFrame({"train_p_value": train, "val_p_value": val})


def test_qval_comparison_empty_does_not_crash(tmp_path):
    # Regression: empty input previously hit ZeroDivisionError in multipletests.
    out = tmp_path / "q.png"
    plot_qval_comparison(_df([], []), out)
    assert out.exists()


def test_pval_comparison_empty_does_not_crash(tmp_path):
    out = tmp_path / "p.png"
    plot_pval_comparison(_df([], []), out)
    assert out.exists()


def test_single_point_writes_placeholder(tmp_path):
    out = tmp_path / "q.png"
    plot_qval_comparison(_df([0.01], [0.2]), out)
    assert out.exists()


def test_enough_points_still_plots(tmp_path):
    # Distinct, well-spread points so OLS/LOESS/KDE are all well-defined.
    rng = np.random.default_rng(0)
    train = np.linspace(0.02, 0.98, 12)
    val = np.clip(train + rng.normal(0, 0.05, train.size), 0.001, 0.999)
    out = tmp_path / "q.png"
    plot_qval_comparison(_df(train, val), out)
    assert out.exists()


def test_missing_val_rows_are_rugged_not_dropped(tmp_path):
    # Some hypotheses have a train p-value but no val (val execution failed); they
    # should be reflected on the plot rather than silently dropped.
    rng = np.random.default_rng(1)
    train = list(np.linspace(0.02, 0.98, 10))
    val = list(np.clip(np.array(train) + rng.normal(0, 0.05, 10), 0.001, 0.999))
    train += [0.1, 0.5, 0.9]
    val += [np.nan, np.nan, np.nan]
    for fn, out in [(plot_pval_comparison, "p.png"), (plot_qval_comparison, "q.png")]:
        path = tmp_path / out
        fn(_df(train, val), path)
        assert path.exists()


def test_placeholder_reports_missing_count(tmp_path, monkeypatch):
    import geryon.legacy.eval.plot as plotmod

    captured = {}
    monkeypatch.setattr(
        plotmod, "_save_placeholder", lambda p, msg: captured.update(msg=msg)
    )
    plot_pval_comparison(_df([0.01, 0.2], [np.nan, np.nan]), tmp_path / "p.png")
    assert "2 with val missing" in captured["msg"]
