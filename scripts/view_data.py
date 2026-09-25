#!/usr/bin/env python3
"""Launch harlequin TUI to view ETL output data."""

import argparse
from pathlib import Path
import subprocess
import tempfile

import duckdb

from geryon.codeflow.runner import get_latest_etl_output


def launch_viewer(data_dir: Path) -> None:
    """Launch harlequin TUI to view parquet files.

    Parameters
    ----------
    data_dir : Path
        Directory containing parquet files
    """
    # mktemp returns a path without creating the file; DuckDB has to create it.
    db_path = tempfile.mktemp(suffix=".duckdb")

    conn = duckdb.connect(db_path)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    registered = 0
    for pf in parquet_files:
        table_name = pf.stem
        if not table_name.startswith("meta_"):
            conn.execute(f"CREATE VIEW \"{table_name}\" AS SELECT * FROM '{pf}'")
            registered += 1

    conn.close()

    print(f"Launching harlequin TUI for: {data_dir}")
    print(f"Registered {registered} tables")
    print()
    print("Harlequin keyboard shortcuts:")
    print("  Ctrl+Q: Quit")
    print("  F2: Focus query editor")
    print("  F5: Run query")
    print("  F6: Focus results")
    print()

    subprocess.run(["harlequin", db_path])


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Launch data viewer for ETL output")

    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        help="ETL output directory (default: latest dated version)",
    )
    parser.add_argument(
        "--data-base",
        type=Path,
        default=Path.home() / "data" / "geryon_data",
        help="Base directory for ETL outputs (default: ~/data/geryon_data)",
    )

    args = parser.parse_args()

    if args.data_dir is None:
        data_dir = get_latest_etl_output(args.data_base)
    else:
        data_dir = args.data_dir

    launch_viewer(data_dir)


if __name__ == "__main__":
    main()
