"""
Partition an ETL parquet directory into per-split subdirectories.

Given a directory of ETL parquet files plus ``patient_split.parquet``
(PATIENT_ID -> split), this writes one subdirectory per split (e.g. ``explore``,
``validation``) containing the same tables filtered to that split's patients,
their regenerated ``*.profile.json`` sidecars, the metadata tables copied
verbatim, and a ``SPLIT`` marker file naming the split.

This is what makes the holdout airtight: the inner loop's session points at the
``explore`` subdir, so the validation cohort is physically absent from the DuckDB
views and the read-only sandbox mount — there is no per-query chokepoint to trust.

Most tables carry ``PATIENT_ID`` directly. Sample-keyed tables (mutations, SVs,
gene matrix) reach a patient only through ``clinical_sample`` (SAMPLE_ID ->
PATIENT_ID); they are filtered by the set of SAMPLE_IDs belonging to the split's
patients. A sample maps to exactly one patient, so the sample-level split follows
the patient-level split with no leakage. Samples absent from ``clinical_sample``
(orphans) are dropped from every split. Metadata tables with no patient key are
copied unchanged.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil

import duckdb
import pandas as pd  # type: ignore

from geryon.etl.profiler import profile_parquet
from geryon.etl.writers import write_parquet, write_profile

logger = logging.getLogger(__name__)

SPLIT_MARKER_FILENAME = "SPLIT"
SPLIT_TABLE_FILENAME = "patient_split.parquet"
CLINICAL_SAMPLE_FILENAME = "data_clinical_sample.parquet"

# Canonical split names (must match create_patient_split's column values).
EXPLORE_SPLIT = "explore"
VALIDATION_SPLIT = "validation"


def read_split_marker(parquet_dir: Path | str) -> str | None:
    """Return the split name a parquet dir was filtered to, or None if unmarked."""
    marker = Path(parquet_dir) / SPLIT_MARKER_FILENAME
    return marker.read_text().strip() if marker.exists() else None


# Sample-keyed tables: the value maps the table to the column holding a SAMPLE_ID,
# which reaches a patient only through clinical_sample. Checked BEFORE the generic
# PATIENT_ID rule, because CNA's column is *named* PATIENT_ID but actually holds
# sample barcodes (e.g. "P-0080859-T01-IM7") — every value is a clinical_sample
# SAMPLE_ID, none a real patient id. The legacy executor keyed it the same way.
#
# Any table not listed here and lacking a real PATIENT_ID column is metadata and
# copied verbatim.
SAMPLE_KEY_COLUMNS: dict[str, str] = {
    "data_CNA.parquet": "PATIENT_ID",
    "data_gene_matrix.parquet": "SAMPLE_ID",
    "data_mutations_extended.parquet": "Tumor_Sample_Barcode",
    "data_mutations_manual.parquet": "Tumor_Sample_Barcode",
    "data_nonsignedout_mutations.parquet": "Tumor_Sample_Barcode",
    "data_sv.parquet": "Sample_ID",
}

PATIENT_ID_COLUMN = "PATIENT_ID"


def _parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    return [r[0] for r in rows]


def _split_patient_ids(split_table: Path) -> dict[str, set[str]]:
    """Map each split name to its set of PATIENT_IDs."""
    df = pd.read_parquet(split_table)
    return {
        str(name): set(group[PATIENT_ID_COLUMN].astype(str))
        for name, group in df.groupby("split")
    }


def _split_sample_ids(clinical_sample: Path, patient_ids: set[str]) -> set[str]:
    """SAMPLE_IDs whose patient is in ``patient_ids`` (from clinical_sample)."""
    df = pd.read_parquet(clinical_sample, columns=["SAMPLE_ID", PATIENT_ID_COLUMN])
    keep = df[df[PATIENT_ID_COLUMN].astype(str).isin(patient_ids)]
    return set(keep["SAMPLE_ID"].astype(str))


def _write_filtered(
    con: duckdb.DuckDBPyConnection,
    src: Path,
    dest: Path,
    where_col: str,
    keep_ids: set[str],
) -> None:
    """Write rows of ``src`` whose ``where_col`` is in ``keep_ids`` to ``dest``."""
    df = con.execute(
        f"SELECT * FROM read_parquet('{src}') "
        f'WHERE CAST("{where_col}" AS VARCHAR) IN '
        f"(SELECT id FROM keep_ids)"
    ).df()
    write_parquet(df, dest)


def split_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    split_table: Path | str | None = None,
) -> list[Path]:
    """Partition ``input_dir`` into one subdir of ``output_dir`` per split.

    Parameters
    ----------
    input_dir
        Directory of ETL parquet files (with ``patient_split.parquet``).
    output_dir
        Parent directory; each split becomes a subdir (e.g. ``explore``).
    split_table
        Path to the split parquet. Defaults to ``input_dir/patient_split.parquet``.

    Returns
    -------
    list[Path]
        The split subdirectories that were written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    split_table = (
        Path(split_table)
        if split_table is not None
        else input_dir / SPLIT_TABLE_FILENAME
    )

    if not split_table.exists():
        raise FileNotFoundError(f"Split table not found: {split_table}")

    clinical_sample = input_dir / CLINICAL_SAMPLE_FILENAME
    if not clinical_sample.exists():
        raise FileNotFoundError(
            f"{CLINICAL_SAMPLE_FILENAME} required to resolve sample-keyed tables"
        )

    ids_by_split = _split_patient_ids(split_table)
    parquet_files = sorted(p for p in input_dir.glob("*.parquet"))

    con = duckdb.connect(":memory:")
    written: list[Path] = []
    try:
        for split_name, patient_ids in ids_by_split.items():
            sample_ids = _split_sample_ids(clinical_sample, patient_ids)
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for src in parquet_files:
                if src.name == SPLIT_TABLE_FILENAME:
                    continue  # the master split table is not copied into a split

                dest = split_dir / src.name
                cols = _parquet_columns(con, src)

                if src.name in SAMPLE_KEY_COLUMNS:
                    where_col = SAMPLE_KEY_COLUMNS[src.name]
                    keep_ids = sample_ids
                elif PATIENT_ID_COLUMN in cols:
                    where_col = PATIENT_ID_COLUMN
                    keep_ids = patient_ids
                else:
                    shutil.copyfile(src, dest)  # metadata: copy verbatim
                    src_profile = src.with_suffix(".profile.json")
                    if src_profile.exists():
                        shutil.copyfile(src_profile, dest.with_suffix(".profile.json"))
                    continue

                con.register("keep_ids", pd.DataFrame({"id": sorted(keep_ids)}))
                _write_filtered(con, src, dest, where_col, keep_ids)
                con.unregister("keep_ids")

                profile = profile_parquet(dest)
                write_profile(dict(profile), dest.with_suffix(".profile.json"))

            (split_dir / SPLIT_MARKER_FILENAME).write_text(split_name + "\n")
            logger.info(
                "Wrote split '%s' (%d patients) to %s",
                split_name,
                len(patient_ids),
                split_dir,
            )
            written.append(split_dir)
    finally:
        con.close()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split an ETL parquet directory into per-split subdirectories "
        "(exploration vs validation), so the holdout is enforced at the data layer."
    )
    parser.add_argument(
        "--input", required=True, help="ETL parquet directory (with patient_split)"
    )
    parser.add_argument(
        "--output", required=True, help="Parent dir for the per-split subdirectories"
    )
    parser.add_argument(
        "--split-table",
        default=None,
        help="Path to patient_split.parquet (default: <input>/patient_split.parquet)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    split_directory(args.input, args.output, args.split_table)


if __name__ == "__main__":
    main()
