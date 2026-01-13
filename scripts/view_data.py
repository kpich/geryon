#!/usr/bin/env python3
"""Launch datasette to view ETL output data."""

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

    # Get all subdirectories
    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    # Return last one (YYYY-MM-DD format sorts chronologically)
    return subdirs[-1]


def launch_viewer(data_dir: Path, port: int = 8001) -> None:
    """Launch datasette to view parquet files.

    Parameters
    ----------
    data_dir : Path
        Directory containing parquet files
    port : int
        Port for datasette server (default: 8001)
    """
    # Create temporary DuckDB database with views to parquet files
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    conn = duckdb.connect(db_path)

    # Register all parquet files as views
    parquet_files = sorted(data_dir.glob("*.parquet"))
    registered = 0
    for pf in parquet_files:
        table_name = pf.stem  # Remove .parquet extension
        # Skip meta_ files (just metadata)
        if not table_name.startswith("meta_"):
            conn.execute(f"CREATE VIEW \"{table_name}\" AS SELECT * FROM '{pf}'")
            registered += 1

    conn.close()

    print(f"Launching datasette for: {data_dir}")
    print(f"Registered {registered} tables")
    print(f"Opening browser at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print()

    # Launch datasette
    subprocess.run(
        [
            "datasette",
            db_path,
            "--port",
            str(port),
            "--open",  # Auto-open browser
        ]
    )


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
        default=Path.home() / "data" / "msk_cycle_data",
        help="Base directory for ETL outputs (default: ~/data/msk_cycle_data)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port for datasette server (default: 8001)",
    )

    args = parser.parse_args()

    if args.data_dir is None:
        data_dir = get_latest_etl_output(args.data_base)
    else:
        data_dir = args.data_dir

    launch_viewer(data_dir, args.port)


if __name__ == "__main__":
    main()
