"""Convert one cBioPortal TSV file to parquet plus a column-profile sidecar."""

import argparse
import logging
from pathlib import Path
import sys
from typing import Literal

from geryon.etl.profiler import profile_parquet
from geryon.etl.readers import read_tsv, write_cna_matrix_to_parquet
from geryon.etl.writers import write_parquet, write_profile

logger = logging.getLogger(__name__)


def _write_profile(parquet_path: Path) -> None:
    """Profile a parquet file and write the sidecar JSON."""
    logger.info("Profiling columns...")
    profile = profile_parquet(parquet_path)
    profile_path = parquet_path.with_suffix(".profile.json")
    write_profile(dict(profile), profile_path)
    logger.info(f"Wrote profile: {profile_path.name}")


def process_cbioportal_file(
    input_path: Path,
    output_path: Path,
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] = "snappy",
) -> None:
    """Convert a cBioPortal TSV file to parquet.

    The CNA file is special-cased: it arrives with one column per sample and is
    transposed to one row per sample, one column per gene.
    """
    if "data_CNA" in input_path.name:  # not meta_CNA
        logger.info(f"Reading CNA file: {input_path}")
        logger.info("Transposing to one row per sample")
        write_cna_matrix_to_parquet(input_path, output_path)
        _write_profile(output_path)
        logger.info("Success!")
        return

    logger.info(f"Reading TSV file: {input_path}")
    df = read_tsv(input_path)

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    logger.info(f"Writing to parquet: {output_path}")
    write_parquet(df, output_path, compression=compression)
    _write_profile(output_path)

    logger.info("Success!")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert cBioPortal TSV file to parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m geryon.etl.process_cbioportal_file \\
      --input data.txt --output data.parquet
  python -m geryon.etl.process_cbioportal_file \\
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
