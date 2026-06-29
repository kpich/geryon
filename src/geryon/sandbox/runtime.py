"""In-container runtime API available to every sandboxed script.

This module is baked into the Docker image and imported as ``geryon_runtime``. It
is deliberately **standalone** — it must not import anything from the ``geryon``
package, because only this single file (plus third-party deps) lives in the image.

Scripts use it like::

    from geryon_runtime import db, report

    con = db()
    df = con.execute("SELECT OS_MONTHS, OS_STATUS FROM clinical_patient").df()
    # ... compute a hazard ratio ...
    report(effect_size=hr, effect_size_type="hazard_ratio", p_value=p,
           ci=(lo, hi), n_a=na, n_b=nb, summary="HR 1.8 for TP53-mut")

``db()`` returns an in-memory DuckDB connection with the parquet files in
``$DATA_DIR`` (default ``/data``) registered as views. Scripts may freely
``CREATE TABLE`` / ``INSERT`` — those land in the ephemeral in-memory catalog and
cannot mutate the source parquet (which is also mounted read-only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb

DATA_DIR = Path(os.environ.get("GERYON_DATA_DIR", "/data"))
SCRATCH_DIR = Path(os.environ.get("GERYON_SCRATCH_DIR", "/scratch"))
RESULT_PATH = SCRATCH_DIR / "result.json"


def _table_name(parquet_path: Path) -> str:
    """Mirror geryon.db.database._get_table_name (strip a leading 'data_')."""
    stem = parquet_path.stem
    return stem[5:] if stem.startswith("data_") else stem


def db() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB with parquet in $DATA_DIR registered as views."""
    con = duckdb.connect(":memory:")
    for parquet_file in sorted(DATA_DIR.glob("*.parquet")):
        name = _table_name(parquet_file)
        con.execute(f"CREATE VIEW \"{name}\" AS SELECT * FROM '{parquet_file}'")
    return con


def artifact_path(name: str) -> Path:
    """Return a path under the scratch dir for writing an artifact (e.g. a plot)."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    return SCRATCH_DIR / name


def report(
    *,
    effect_size: float | None = None,
    effect_size_type: str | None = None,
    p_value: float | None = None,
    ci: tuple[float | None, float | None] | None = None,
    n_a: int | None = None,
    n_b: int | None = None,
    summary: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the standardized result to $SCRATCH_DIR/result.json.

    Last call wins (idempotent). All fields are optional — report whatever the
    analysis produced. ``ci`` is a ``(lower, upper)`` tuple.
    """
    ci_lower, ci_upper = (None, None)
    if ci is not None:
        ci_lower, ci_upper = ci

    payload: dict[str, Any] = {
        "effect_size": _as_float(effect_size),
        "effect_size_type": effect_size_type,
        "p_value": _as_float(p_value),
        "ci_lower": _as_float(ci_lower),
        "ci_upper": _as_float(ci_upper),
        "n_a": _as_int(n_a),
        "n_b": _as_int(n_b),
        "summary": summary,
        "extra": {k: _as_float(v) for k, v in (extra or {}).items()},
    }

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload))


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_int(value: Any) -> int | None:
    return None if value is None else int(value)
