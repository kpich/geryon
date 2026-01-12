"""
Data readers for ETL pipeline.

Handles reading TSV files with various formats from MSK-IMPACT datasets.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def read_tsv(
    file_path: str | Path,
    *,
    sep: str = "\t",
    comment: str = "#",
    **kwargs: Any,
) -> pd.DataFrame:
    """Read TSV file with comment line handling and low_memory=False for mixed types."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read TSV with standard parameters
    # Use low_memory=False to avoid dtype warnings on large files with mixed types
    df = pd.read_csv(
        file_path,
        sep=sep,
        comment=comment,
        low_memory=False,
        **kwargs,
    )

    return df


def write_cna_matrix_to_parquet(
    input_path: str | Path, output_path: str | Path
) -> None:
    """Write CNA wide matrix directly to parquet in long format using DuckDB.

    Bypasses pandas to avoid materializing 110M rows in memory.
    Transforms directly from wide TSV to long-format parquet.

    Parameters
    ----------
    input_path : str | Path
        Path to CNA TSV file (wide format)
    output_path : str | Path
        Path to output parquet file (long format)
    """
    import duckdb

    input_path = Path(input_path)
    output_path = Path(output_path)

    conn = duckdb.connect(":memory:")

    # Read TSV into DuckDB (handles 156k columns efficiently)
    # Increase max_line_size to handle 156k+ columns (~3MB per line)
    conn.execute(f"""
        CREATE TABLE cna_wide AS
        SELECT * FROM read_csv_auto(
            '{input_path}',
            delim='\t',
            header=true,
            max_line_size=10000000
        )
    """)

    # Get the first column name (gene identifier)
    gene_col = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='cna_wide' LIMIT 1"
    ).fetchone()[0]

    # UNPIVOT to long format and write directly to parquet
    # DuckDB streams this without materializing in memory
    conn.execute(f"""
        COPY (
            UNPIVOT cna_wide
            ON COLUMNS(* EXCLUDE ("{gene_col}"))
            INTO
                NAME patient_id
                VALUE cna_value
        ) TO '{output_path}' (FORMAT PARQUET)
    """)

    conn.close()


def read_cna_matrix(file_path: str | Path) -> pd.DataFrame:
    """Read CNA wide matrix and convert to long format.

    Transforms from wide format (one column per patient sample):
        Hugo_Symbol | P-001-T01 | P-002-T01 | ... (156,450+ patient columns)
        KRAS        | 0         | 1         | ...
        TP53        | -1        | 0         | ...

    To long format:
        patient_id | gene  | cna_value
        P-001-T01  | KRAS  | 0
        P-001-T01  | TP53  | -1
        P-002-T01  | KRAS  | 1
        P-002-T01  | TP53  | 0

    This is the standard relational database format that enables:
    - Fast schema discovery (3 columns instead of 156,451)
    - SQL queries: SELECT * FROM cna WHERE patient_id = 'X' AND gene = 'KRAS'
    - Reasonable LLM context (can describe 3 columns vs 156k)

    Uses DuckDB for efficient memory usage - can UNPIVOT 156k columns without
    loading 110M rows into pandas memory.

    Parameters
    ----------
    file_path : str | Path
        Path to CNA TSV file

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: patient_id, gene, cna_value
    """
    import duckdb

    file_path = Path(file_path)

    # Use DuckDB to efficiently UNPIVOT without loading full result into memory
    conn = duckdb.connect(":memory:")

    # Read TSV into DuckDB (handles 156k columns efficiently)
    # Increase max_line_size to handle 156k+ columns (~3MB per line)
    conn.execute(f"""
        CREATE TABLE cna_wide AS
        SELECT * FROM read_csv_auto(
            '{file_path}',
            delim='\t',
            header=true,
            max_line_size=10000000
        )
    """)

    # Get the first column name (gene identifier)
    gene_col = conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='cna_wide' LIMIT 1"
    ).fetchone()[0]

    # UNPIVOT to long format - DuckDB streams this efficiently
    # Quote column name to handle special characters (colons, spaces, etc)
    result = conn.execute(f"""
        UNPIVOT cna_wide
        ON COLUMNS(* EXCLUDE ("{gene_col}"))
        INTO
            NAME patient_id
            VALUE cna_value
    """).df()

    # Rename gene column and reorder
    result = result.rename(columns={gene_col: "gene"})
    result = result[["patient_id", "gene", "cna_value"]]

    conn.close()

    return result


def get_table_name(file_path: str | Path) -> str:
    """Extract table name from filename, removing 'data_' prefix if present."""
    file_path = Path(file_path)
    stem = file_path.stem

    # Remove 'data_' prefix if present
    if stem.startswith("data_"):
        return stem[5:]

    return stem
