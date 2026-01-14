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
from typing import Literal

from msk_cycl.etl.readers import read_tsv, write_cna_matrix_to_parquet
from msk_cycl.etl.writers import write_parquet

logger = logging.getLogger(__name__)


def process_cbioportal_file(
    input_path: Path,
    output_path: Path,
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
) -> None:
    """Convert cBioPortal TSV file to parquet format.

    Special handling for CNA files: transposes wide matrix (1 column per patient)
    to long format (patient_id, gene, cna_value) for efficient SQL querying.
    Uses DuckDB to stream directly to parquet without materializing 110M rows.
    """
    # Use specialized writer for CNA data files (wide matrix → long format)
    # Match data_CNA but not meta_CNA
    # Bypass pandas entirely to avoid memory issues with 110M rows
    if "data_CNA" in input_path.name:
        logger.info(f"Reading CNA file: {input_path}")
        logger.info("Transposing wide matrix to long format (streaming to parquet)")
        write_cna_matrix_to_parquet(input_path, output_path)
        logger.info("Success!")
        return

    # Standard flow for non-CNA files
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

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    process_cbioportal_file(
        input_path=args.input,
        output_path=args.output,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
