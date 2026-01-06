"""
CLI entry point for processing cBioPortal TSV files.

This module provides a command-line interface for converting cBioPortal-formatted
TSV files to parquet. It is designed to be invoked via
`python -m msk_cycl.etl.process_cbioportal_file`.
"""

import argparse
import logging
from pathlib import Path
import sys

from msk_cycl.etl.readers import read_tsv
from msk_cycl.etl.writers import write_parquet

logger = logging.getLogger(__name__)


def process_cbioportal_file(
    input_path: Path,
    output_path: Path,
    compression: str = "snappy",
) -> None:
    """Convert cBioPortal TSV file to parquet format."""
    logger.info(f"Reading TSV file: {input_path}")
    df = read_tsv(input_path)

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    logger.info(f"Writing to parquet: {output_path}")
    write_parquet(df, output_path, compression=compression)

    logger.info("Success!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert cBioPortal TSV file to parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m msk_cycl.etl.process_cbioportal_file \\
      --input data.txt --output data.parquet
  python -m msk_cycl.etl.process_cbioportal_file \\
      -i data.txt -o data.parquet --log-level DEBUG
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    # Process file
    process_cbioportal_file(
        input_path=args.input,
        output_path=args.output,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
