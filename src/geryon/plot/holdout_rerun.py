"""Rerun every successful hypothesis script, unchanged, on the validation split.

The scripts read whatever is mounted at ``/data``, so mounting the session's
``validation/`` dir in place of ``explore/`` is the whole rerun. No LLM is involved.
Writes one JSON line per hypothesis. A script that fails on the held-out patients is
a result, not an error. A sandbox problem (docker down, image missing) raises.

The output holds held-out results. Keep it out of the sessions dir: anything under
there can reach the agents' prompts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from geryon.codeflow.models import CodeHypothesis
from geryon.etl.data_version import resolve_data_version
from geryon.etl.split_by_patient import (
    EXPLORE_SPLIT,
    VALIDATION_SPLIT,
    read_split_marker,
)
from geryon.plot._critiques import load_hypotheses
from geryon.sandbox.runner import DEFAULT_TIMEOUT_SECONDS, run_script

_STDERR_TAIL = 2000


def validation_dir(parquet_dir: Path, data_version: str | None) -> Path:
    """The validation split beside the explore split a session read."""
    if read_split_marker(parquet_dir) != EXPLORE_SPLIT:
        raise ValueError(f"session parquet_dir is not an explore split: {parquet_dir}")
    val = parquet_dir.parent / VALIDATION_SPLIT
    if read_split_marker(val) != VALIDATION_SPLIT:
        raise ValueError(f"no validation split beside {parquet_dir}")
    found = resolve_data_version(val)
    if data_version is not None and found != data_version:
        raise ValueError(
            f"{val} is data version {found!r}, the hypothesis ran on {data_version!r}"
        )
    return val


def session_configs(data_dir: Path) -> dict[str, dict]:
    configs = {}
    for path in data_dir.rglob("config.json"):
        cfg = json.loads(path.read_text())
        if "session_id" in cfg:
            configs[cfg["session_id"]] = cfg
    return configs


def rerun(h: CodeHypothesis, cfg: dict) -> dict:
    val = validation_dir(Path(cfg["parquet_dir"]), h.data_version)
    run = run_script(
        h.code,
        val,
        timeout=cfg.get("sandbox_timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
    )
    return {
        "hypothesis_id": h.hypothesis_id,
        "validation_dir": str(val),
        "success": run.success,
        "timed_out": run.timed_out,
        "error": run.error,
        "stderr_tail": run.stderr[-_STDERR_TAIL:],
        "result": run.result.model_dump() if run.result else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chain", default=None, help="Only this chain (default: all)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    configs = session_configs(args.data_dir)
    hyps = [
        h
        for h in load_hypotheses(args.data_dir)
        if h.success and (args.chain is None or h.chain == args.chain)
    ]
    missing = {h.session_id for h in hyps} - configs.keys()
    if missing:
        raise ValueError(f"no config.json for session(s) {sorted(missing)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda h: rerun(h, configs[h.session_id]), hyps))
    with open(args.output, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ok = sum(r["result"] is not None for r in rows)
    print(f"reran {len(rows)} scripts on validation; {ok} reported a result")


if __name__ == "__main__":
    main()
