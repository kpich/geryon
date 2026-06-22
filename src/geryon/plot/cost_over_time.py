"""Plot cumulative token usage and dollar cost over the run.

Reads generation_usage events from each session's trace.jsonl. Caching is not
enabled in the generation loop, so cost is exact:
input * $5/1M + output * $25/1M.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# USD per token (Anthropic pricing). input_tokens is the TOTAL input; the cached
# portion (cache_read / cache_creation) is billed at its own rate and the
# remainder at the full input rate.
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
                    events.append(rec)
    events.sort(key=lambda r: r.get("ts", ""))
    return events


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
        inp = np.array([e.get("input_tokens", 0) for e in events], dtype=float)
        out = np.array([e.get("output_tokens", 0) for e in events], dtype=float)
        cache_read = np.array(
            [e.get("cache_read_tokens", 0) for e in events], dtype=float
        )
        cache_write = np.array(
            [e.get("cache_creation_tokens", 0) for e in events], dtype=float
        )

        # input_tokens is the total input; split out the cached components, which
        # are billed at their own rates. Old runs have no cache fields (-> 0), so
        # this reduces to input*$5 + output*$25.
        uncached_inp = np.clip(inp - cache_read - cache_write, 0, None)
        cost = (
            uncached_inp * _INPUT_PRICE
            + cache_read * _CACHE_READ_PRICE
            + cache_write * _CACHE_WRITE_PRICE
            + out * _OUTPUT_PRICE
        )

        cum_inp = np.cumsum(inp)
        cum_out = np.cumsum(out)
        cum_cost = np.cumsum(cost)

        ax_cost.plot(x, cum_cost, color="#0072B2", linewidth=1.8)
        ax_cost.fill_between(x, cum_cost, color="#0072B2", alpha=0.15)
        ax_cost.text(
            0.98,
            0.05,
            f"total: ${cum_cost[-1]:,.2f}",
            transform=ax_cost.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#0072B2",
        )

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
