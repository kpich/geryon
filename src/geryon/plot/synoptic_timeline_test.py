"""Tests for synoptic_timeline helpers."""

from geryon.plot.synoptic_timeline import (
    _pooled_cache_fractions,
    _uncached_cost,
    iteration_index,
)


def test_iteration_index_global_sequence_across_sessions():
    events = [
        {"_session": "geryon_data/sessions/d/AAA/trace.jsonl", "iteration": 1},
        {"_session": "geryon_data/sessions/d/AAA/trace.jsonl", "iteration": 2},
        {"_session": "geryon_data/sessions/d/BBB/trace.jsonl", "iteration": 1},
    ]
    itmap = iteration_index(events)
    assert itmap[("AAA", 1)] == 1
    assert itmap[("AAA", 2)] == 2
    assert itmap[("BBB", 1)] == 3  # global index continues across sessions


def test_uncached_cost_prices_full_volume_at_input_rate():
    e = {
        "input_tokens": 100,
        "cache_read_tokens": 800,
        "cache_creation_tokens": 100,
        "output_tokens": 50,
    }
    # full prompt volume (1000) at $5/1M + output at $25/1M, ignoring any caching
    expected = (1000 * 5.0 + 50 * 25.0) / 1_000_000
    assert abs(_uncached_cost(e) - expected) < 1e-12


def test_pooled_cache_fractions_volume_weighted_across_runs():
    events = [
        {"input_tokens": 300, "cache_read_tokens": 600, "cache_creation_tokens": 100},
        {"input_tokens": 10, "cache_read_tokens": 950, "cache_creation_tokens": 40},
    ]
    total = 1000 + 1000
    f_read, f_write = _pooled_cache_fractions(events)
    assert abs(f_read - (600 + 950) / total) < 1e-12
    assert abs(f_write - (100 + 40) / total) < 1e-12


def test_pooled_cache_fractions_none_without_cached_runs():
    events = [{"input_tokens": 100, "cache_read_tokens": 0}]
    assert _pooled_cache_fractions(events) is None
