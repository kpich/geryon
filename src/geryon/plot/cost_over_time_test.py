"""Tests for cost_over_time pricing and cache-fraction calibration."""

from geryon.plot.cost_over_time import (
    _cache_fractions,
    _estimated_cost,
    _event_cost,
    _input_volume,
)


def test_event_cost_disjoint_buckets():
    # input/cache_read/cache_write are disjoint; each billed at its own rate.
    e = {
        "input_tokens": 1000,
        "cache_read_tokens": 2000,
        "cache_creation_tokens": 500,
        "output_tokens": 100,
    }
    expected = (1000 * 5.0 + 2000 * 0.5 + 500 * 6.25 + 100 * 25.0) / 1_000_000
    assert abs(_event_cost(e) - expected) < 1e-12


def test_event_cost_uncached_legacy_event():
    e = {"input_tokens": 1000, "output_tokens": 100}
    expected = (1000 * 5.0 + 100 * 25.0) / 1_000_000
    assert abs(_event_cost(e) - expected) < 1e-12


def test_input_volume_sums_three_buckets():
    e = {"input_tokens": 25, "cache_read_tokens": 700, "cache_creation_tokens": 38}
    assert _input_volume(e) == 763


def test_cache_fractions_uses_latest_session_only():
    events = [
        # older, low-hit session
        {
            "_session": "s1",
            "ts": "2026-01-01T00:00:00Z",
            "input_tokens": 500,
            "cache_read_tokens": 500,
            "cache_creation_tokens": 0,
        },
        # newest session: ~95% cache read
        {
            "_session": "s2",
            "ts": "2026-02-01T00:00:00Z",
            "input_tokens": 10,
            "cache_read_tokens": 950,
            "cache_creation_tokens": 40,
        },
    ]
    f_read, f_write = _cache_fractions(events)
    assert abs(f_read - 0.95) < 1e-9
    assert abs(f_write - 0.04) < 1e-9


def test_cache_fractions_none_without_cached_runs():
    assert _cache_fractions([{"input_tokens": 100, "cache_read_tokens": 0}]) is None


def test_estimated_cost_applies_fractions_to_uncached_event():
    # uncached old event of 1000 input tokens, at 90% read / 5% write
    e = {"input_tokens": 1000, "output_tokens": 100}
    cost = _estimated_cost(e, (0.9, 0.05))
    expected = (50 * 5.0 + 900 * 0.5 + 50 * 6.25 + 100 * 25.0) / 1_000_000
    assert abs(cost - expected) < 1e-12
