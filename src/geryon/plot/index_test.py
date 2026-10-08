"""Tests for the plot index page."""

import os

import pytest

from geryon.plot.index import build


def test_lists_only_plots_from_this_run(tmp_path):
    (tmp_path / "cost").mkdir()
    (tmp_path / "OLD").mkdir()
    fresh = tmp_path / "cost" / "cost_over_time.svg"
    stale = tmp_path / "OLD" / "qval_by_trust.svg"
    fresh.write_bytes(b"")
    stale.write_bytes(b"")
    os.utime(stale, (1000, 1000))

    page = build(tmp_path, since=2000)
    assert 'src="cost/cost_over_time.svg"' in page
    assert 'href="cost/cost_over_time.pdf"' in page
    assert "qval_by_trust" not in page and "OLD" not in page


def test_no_fresh_plots_fails(tmp_path):
    (tmp_path / "a.svg").write_bytes(b"")
    os.utime(tmp_path / "a.svg", (1000, 1000))
    with pytest.raises(SystemExit):
        build(tmp_path, since=2000)
