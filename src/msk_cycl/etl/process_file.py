"""
CLI entry point for processing individual TSV files.

This module provides a command-line interface for converting TSV files to parquet.
It is designed to be invoked via `python -m msk_cycl.etl.process_file`.
"""

import argparse
from pathlib import Path
import sys

from msk_cycl.etl.readers import read_tsv
from msk_cycl.etl.writers import write_parquet


def main() -> int:
    """
    Main entry point for the process_file CLI.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Convert TSV file to parquet format for MSK CYCL ETL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m msk_cycl.etl.process_file --input data.txt --output data.parquet
  python -m msk_cycl.etl.process_file -i data.txt -o data.parquet
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to input TSV file",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Path to output parquet file",
    )

    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "lz4", "zstd"],
        help="Compression algorithm for parquet (default: snappy)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        if args.verbose:
            print(f"Reading TSV file: {args.input}")

        df = read_tsv(args.input)

        if args.verbose:
            print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            print(f"Writing to parquet: {args.output}")

        write_parquet(df, args.output, compression=args.compression)

        if args.verbose:
            print("Success!")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
