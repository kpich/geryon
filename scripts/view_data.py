#!/usr/bin/env python3
"""Launch harlequin TUI to view ETL output data."""

import argparse
from pathlib import Path
import subprocess
import tempfile

import duckdb


def get_latest_etl_output(base_dir: Path) -> Path:
    """Find latest ETL output directory by name.

    Parameters
    ----------
    base_dir : Path
        Base directory containing timestamped subdirectories

    Returns
    -------
    Path
        Latest subdirectory (by name sort)

    Raises
    ------
    FileNotFoundError
        If no subdirectories found or base_dir doesn't exist
    """
    base_path = base_dir.expanduser()

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_path}")

    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    # Return last one (YYYY-MM-DD format sorts chronologically)
    return subdirs[-1]


def launch_viewer(data_dir: Path) -> None:
    """Launch harlequin TUI to view parquet files.

    Parameters
    ----------
    data_dir : Path
        Directory containing parquet files
    """
    # Create temporary DuckDB database with views to parquet files
    # Use mktemp to get path without creating file (DuckDB needs to create it)
    db_path = tempfile.mktemp(suffix=".duckdb")

    conn = duckdb.connect(db_path)

    parquet_files = sorted(data_dir.glob("*.parquet"))
    registered = 0
    for pf in parquet_files:
        table_name = pf.stem  # Remove .parquet extension
        # Skip meta_ files (just metadata)
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
        help="ETL output directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--data-base",
        type=Path,
        default=Path.home() / "data" / "geryone_data",
        help="Base directory for ETL outputs (default: ~/data/geryone_data)",
    )

    args = parser.parse_args()

    if args.data_dir is None:
        data_dir = get_latest_etl_output(args.data_base)
    else:
        data_dir = args.data_dir

    launch_viewer(data_dir)


if __name__ == "__main__":
    main()
