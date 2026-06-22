"""Plot cumulative token usage and dollar cost over the run.

Reads generation_usage events from each session's trace.jsonl and prices them at
Anthropic rates. For runs predating prompt caching (no cache fields), a dashed
"estimated with caching" line shows what they would have cost, calibrated from
the cache hit-rate observed in the runs that did use caching.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# USD per token (Anthropic pricing). input_tokens, cache_read_tokens and
# cache_creation_tokens are three DISJOINT buckets of the prompt: fresh input is
# billed at 1x, cache reads at 0.1x, cache writes at 1.25x. (total_tokens only
# sums input + output + cache_read, so it understates volume — don't use it.)
_INPUT_PRICE = 5.00 / 1_000_000
_OUTPUT_PRICE = 25.00 / 1_000_000
_CACHE_READ_PRICE = 0.50 / 1_000_000
_CACHE_WRITE_PRICE = 6.25 / 1_000_000  # 5-minute TTL


def load_generation_usage(data_dir: Path) -> list[dict]:
    """Collect generation_usage events across all sessions, ordered by timestamp."""
    events: list[dict] = []
    for trace_file in sorted((data_dir / "sessions").rglob("trace.jsonl")):
        with open(trace_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "generation_usage":
                    rec["_session"] = str(trace_file)
                    events.append(rec)
    events.sort(key=lambda r: r.get("ts", ""))
    return events


def _event_cost(e: dict) -> float:
    """Actual USD cost of one generation_usage event."""
    return (
        e.get("input_tokens", 0) * _INPUT_PRICE
        + e.get("cache_creation_tokens", 0) * _CACHE_WRITE_PRICE
        + e.get("cache_read_tokens", 0) * _CACHE_READ_PRICE
        + e.get("output_tokens", 0) * _OUTPUT_PRICE
    )


def _input_volume(e: dict) -> int:
    """Full prompt volume of an event: the three disjoint input buckets."""
    return (
        e.get("input_tokens", 0)
        + e.get("cache_read_tokens", 0)
        + e.get("cache_creation_tokens", 0)
    )


def _cache_fractions(events: list[dict]) -> tuple[float, float] | None:
    """Calibrate (cache_read, cache_write) fractions of input volume from the
    most recent cached session, so the estimate reflects current caching
    behavior (e.g. tail caching) rather than an average over older regimes.
    Returns None if no cached runs are present.
    """
    cached = [e for e in events if e.get("cache_read_tokens", 0) > 0]
    if not cached:
        return None
    latest_session = max(cached, key=lambda e: e.get("ts", ""))["_session"]
    cached = [e for e in cached if e["_session"] == latest_session]
    volume = sum(_input_volume(e) for e in cached)
    if volume <= 0:
        return None
    f_read = sum(e["cache_read_tokens"] for e in cached) / volume
    f_write = sum(e.get("cache_creation_tokens", 0) for e in cached) / volume
    return f_read, f_write


def _estimated_cost(e: dict, fractions: tuple[float, float]) -> float:
    """Cost of an event as if it had been cached at the calibrated hit-rate.

    Cached events keep their actual cost; uncached events (input_tokens is the
    full prompt volume) have the observed fractions applied.
    """
    if e.get("cache_read_tokens", 0) > 0:
        return _event_cost(e)
    f_read, f_write = fractions
    volume = _input_volume(e)  # uncached run: == input_tokens (full prompt)
    out = e.get("output_tokens", 0)
    read = f_read * volume
    write = f_write * volume
    fresh = max(volume - read - write, 0)
    return (
        fresh * _INPUT_PRICE
        + write * _CACHE_WRITE_PRICE
        + read * _CACHE_READ_PRICE
        + out * _OUTPUT_PRICE
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cumulative token usage and cost over the run"
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    events = load_generation_usage(args.data_dir)

    fig, (ax_cost, ax_tok) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    if events:
        x = np.arange(1, len(events) + 1)
        out = np.array([e.get("output_tokens", 0) for e in events], dtype=float)
        # Full input volume per call: fresh + cache read + cache write.
        volume = np.array([_input_volume(e) for e in events], dtype=float)

        cum_cost = np.cumsum([_event_cost(e) for e in events])
        cum_inp = np.cumsum(volume)
        cum_out = np.cumsum(out)

        ax_cost.plot(x, cum_cost, color="#0072B2", linewidth=1.8, label="actual")
        ax_cost.fill_between(x, cum_cost, color="#0072B2", alpha=0.15)
        ax_cost.text(
            0.98,
            0.05,
            f"actual: ${cum_cost[-1]:,.2f}",
            transform=ax_cost.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#0072B2",
        )

        fractions = _cache_fractions(events)
        any_uncached = any(e.get("cache_read_tokens", 0) == 0 for e in events)
        if fractions is not None and any_uncached:
            cum_est = np.cumsum([_estimated_cost(e, fractions) for e in events])
            ax_cost.plot(
                x,
                cum_est,
                color="#D55E00",
                linewidth=1.5,
                linestyle="--",
                label=f"est. with caching (cache hit {fractions[0]:.0%})",
            )
            ax_cost.text(
                0.98,
                0.16,
                f"est. with caching: ${cum_est[-1]:,.2f}",
                transform=ax_cost.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color="#D55E00",
            )
        ax_cost.legend(loc="upper left", fontsize=9)

        ax_tok.plot(x, cum_inp + cum_out, color="black", linewidth=1.8, label="total")
        ax_tok.plot(x, cum_inp, color="#0072B2", linewidth=1.3, label="input")
        ax_tok.plot(x, cum_out, color="#D55E00", linewidth=1.3, label="output")
        ax_tok.legend(loc="upper left", fontsize=9)
        ax_tok.text(
            0.98,
            0.05,
            f"total: {(cum_inp[-1] + cum_out[-1]):,.0f} tokens",
            transform=ax_tok.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    ax_cost.set_ylabel("Cumulative cost (USD)")
    ax_cost.grid(True, alpha=0.3)
    ax_tok.set_ylabel("Cumulative tokens")
    ax_tok.set_xlabel("Generation iteration")
    ax_tok.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
