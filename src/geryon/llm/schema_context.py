"""Schema context loader — injects verified confounder controls into prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geryon.db import Database


def load_schema_context(parquet_dir: Path, db: Database) -> str:
    """Return a markdown block listing verified confounder control columns.

    Checks for pre-generated schema_context.txt first. Falls back to building
    from confounder_config.json. Returns empty string if neither exists.
    """
    cache_file = parquet_dir / "schema_context.txt"
    if cache_file.exists():
        return cache_file.read_text()

    config_file = parquet_dir / "confounder_config.json"
    if not config_file.exists():
        return ""

    config = json.loads(config_file.read_text())
    columns = config.get("confounder_columns", [])
    if not columns:
        return ""

    return _build_context(db, columns)


def _build_context(db: Database, columns: list[dict]) -> str:
    lines: list[str] = [
        "AVAILABLE CONFOUNDER CONTROLS (verified from this database):",
        "",
    ]

    for col_spec in columns:
        table = col_spec["table"]
        column = col_spec["column"]
        try:
            df = db.execute(
                f'SELECT "{column}", COUNT(*) AS cnt FROM {table} '
                f'GROUP BY "{column}" ORDER BY cnt DESC LIMIT 10'
            )
            if df.empty:
                continue
            values_str = ", ".join(
                f'"{row[column]}" ({row["cnt"]:,})' for _, row in df.iterrows()
            )
            lines.append(f"- {table}.{column}: {values_str}")
        except Exception:
            continue

    lines.append("")
    lines.append(
        "When you see uncontrolled=2 or 3 in prior hypotheses, consider adding one of"
    )
    lines.append("these as a filter to both cohorts to control for the confounder.")

    return "\n".join(lines)


def write_schema_context(parquet_dir: Path, db: Database) -> None:
    """Build schema context and write it to parquet_dir/schema_context.txt."""
    config_file = parquet_dir / "confounder_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No confounder_config.json in {parquet_dir}")

    config = json.loads(config_file.read_text())
    columns = config.get("confounder_columns", [])
    context = _build_context(db, columns)

    out = parquet_dir / "schema_context.txt"
    out.write_text(context)
    print(f"Written: {out}")
    print()
    print(context)


if __name__ == "__main__":
    from geryon.codeflow.runner import get_latest_etl_output

    parser = argparse.ArgumentParser(
        description="Load or generate schema context for confounder controls"
    )
    parser.add_argument(
        "--write", action="store_true", help="Write schema_context.txt to parquet_dir"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Parquet directory (default: auto-detect latest from ~/data/geryon_data)",
    )
    parser.add_argument(
        "--data-base",
        type=Path,
        default=Path.home() / "data" / "geryon_data",
        help="Base directory for ETL outputs (default: ~/data/geryon_data)",
    )
    args = parser.parse_args()

    parquet_dir: Path = args.data_dir or get_latest_etl_output(args.data_base)
    print(f"Using parquet dir: {parquet_dir}")

    db = Database(parquet_dir)

    if args.write:
        write_schema_context(parquet_dir, db)
    else:
        print(load_schema_context(parquet_dir, db))
