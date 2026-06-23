"""Tool for creating persistent derived SQL views."""

import json
from pathlib import Path

from geryon.db import Database

DERIVED_PREFIX = "derived_"


def create_derived_view(db: Database, name: str, sql: str) -> str:
    stripped = sql.strip().upper()
    if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
        return "ERROR: View SQL must be a SELECT or WITH (CTE) query"

    safe_name = name if name.startswith(DERIVED_PREFIX) else f"{DERIVED_PREFIX}{name}"

    etl_tables = {t for t in db.list_tables() if not t.startswith(DERIVED_PREFIX)}
    base_name = safe_name[len(DERIVED_PREFIX) :]
    if base_name in etl_tables:
        return f"ERROR: '{safe_name}' would shadow an existing ETL table"

    try:
        db.create_view(safe_name, sql)
        count_df = db.execute(f'SELECT COUNT(*) AS cnt FROM "{safe_name}"')
        row_count = int(count_df["cnt"].iloc[0])
        desc_df = db.execute(f'DESCRIBE "{safe_name}"')
        cols = desc_df["column_name"].tolist()
        col_str = ", ".join(cols[:10]) + (
            f" ... ({len(cols)} total)" if len(cols) > 10 else ""
        )
        return f"Created view '{safe_name}': {row_count:,} rows, columns: {col_str}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def replay_derived_views(db: Database, views_path: Path) -> list[str]:
    """Recreate the derived views saved in a derived_views.json file into db.

    Views are replayed in file order so that a view depending on an earlier one
    resolves (DuckDB binds view bodies at creation time). Returns the names that
    failed to recreate; an unreadable file is reported as a single failure.
    """
    if not views_path.exists():
        return []
    try:
        defs: dict[str, str] = json.loads(views_path.read_text())
    except Exception:
        return [str(views_path)]
    failed = []
    for name, sql in defs.items():
        try:
            db.create_view(name, sql)
        except Exception:
            failed.append(name)
    return failed
