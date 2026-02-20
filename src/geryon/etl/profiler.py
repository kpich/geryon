"""Column profiling for parquet files.

Precomputes per-column stats (distinct count, top N values, null rate)
so describe_table can show value distributions without expensive runtime queries.
"""

from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import TypedDict

import duckdb

logger = logging.getLogger(__name__)

WIDE_MATRIX_THRESHOLD = 100
LOW_CARDINALITY_THRESHOLD = 20
MAX_VALUE_LENGTH = 60
REPRESENTATIVE_GENE_COUNT = 10


class TopValue(TypedDict):
    value: str
    count: int


class ColumnProfile(TypedDict):
    column_name: str
    dtype: str
    null_count: int
    null_rate: float
    distinct_count: int
    total_count: int
    top_values: list[TopValue]
    is_high_cardinality: bool


class TableProfile(TypedDict, total=False):
    table_name: str
    row_count: int
    column_count: int
    profiled_at: str
    columns: list[ColumnProfile]
    profile_strategy: str
    gene_columns: list[str]


def _get_table_name(parquet_path: Path) -> str:
    stem = parquet_path.stem
    if stem.startswith("data_"):
        return stem[5:]
    return stem


def _profile_column(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    column_name: str,
    dtype: str,
    row_count: int,
    top_n: int,
) -> ColumnProfile:
    """Profile a single column: null count, distinct count, top N values."""
    quoted_col = f'"{column_name}"'

    stats = conn.execute(
        f"SELECT COUNT(*) - COUNT({quoted_col}) AS null_count, "
        f"COUNT(DISTINCT {quoted_col}) AS distinct_count "
        f"FROM read_parquet('{parquet_path}')"
    ).fetchone()
    assert stats is not None
    null_count, distinct_count = int(stats[0]), int(stats[1])

    null_rate = round(null_count / row_count, 4) if row_count > 0 else 0.0

    is_high_cardinality = distinct_count > LOW_CARDINALITY_THRESHOLD

    # For low cardinality, fetch ALL values; for high, fetch top_n
    limit = top_n if is_high_cardinality else distinct_count
    top_rows = conn.execute(
        f"SELECT CAST({quoted_col} AS VARCHAR) AS val, COUNT(*) AS cnt "
        f"FROM read_parquet('{parquet_path}') "
        f"WHERE {quoted_col} IS NOT NULL "
        f"GROUP BY {quoted_col} ORDER BY cnt DESC LIMIT {limit}"
    ).fetchall()

    top_values: list[TopValue] = [
        {"value": str(row[0])[:MAX_VALUE_LENGTH], "count": int(row[1])}
        for row in top_rows
    ]

    return ColumnProfile(
        column_name=column_name,
        dtype=dtype,
        null_count=null_count,
        null_rate=null_rate,
        distinct_count=distinct_count,
        total_count=row_count,
        top_values=top_values,
        is_high_cardinality=is_high_cardinality,
    )


def _detect_strategy(columns: list[tuple[str, str]]) -> str:
    if len(columns) >= WIDE_MATRIX_THRESHOLD:
        return "wide_matrix"
    return "standard"


def profile_parquet(
    path: Path | str,
    top_n: int = 20,
    max_profile_columns: int = 30,
) -> TableProfile:
    """Profile a parquet file, returning column-level stats.

    Parameters
    ----------
    path : Path | str
        Path to parquet file
    top_n : int
        Max top values to include for high-cardinality columns
    max_profile_columns : int
        Max columns to profile for standard strategy
    """
    path = Path(path)
    conn = duckdb.connect(":memory:")

    try:
        row_count_result = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}')"
        ).fetchone()
        assert row_count_result is not None
        row_count = int(row_count_result[0])

        schema = conn.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path}')"
        ).fetchall()
        columns = [(row[0], row[1]) for row in schema]

        strategy = _detect_strategy(columns)
        table_name = _get_table_name(path)

        logger.info(
            f"Profiling {table_name}: {row_count} rows, {len(columns)} cols, "
            f"strategy={strategy}"
        )

        profile = TableProfile(
            table_name=table_name,
            row_count=row_count,
            column_count=len(columns),
            profiled_at=datetime.now(UTC).isoformat(),
            columns=[],
            profile_strategy=strategy,
        )

        if strategy == "wide_matrix":
            # Profile PATIENT_ID (or first col) normally
            non_gene_cols = [
                (name, dtype)
                for name, dtype in columns
                if name.upper() in ("PATIENT_ID", "SAMPLE_ID")
            ]
            gene_cols = [
                (name, dtype)
                for name, dtype in columns
                if name.upper() not in ("PATIENT_ID", "SAMPLE_ID")
            ]

            for col_name, dtype in non_gene_cols:
                col_profile = _profile_column(
                    conn, path, col_name, dtype, row_count, top_n
                )
                profile["columns"].append(col_profile)

            # Profile representative gene columns
            representative = gene_cols[:REPRESENTATIVE_GENE_COUNT]
            for col_name, dtype in representative:
                col_profile = _profile_column(
                    conn, path, col_name, dtype, row_count, top_n
                )
                profile["columns"].append(col_profile)

            profile["gene_columns"] = [name for name, _ in gene_cols]
        else:
            cols_to_profile = columns[:max_profile_columns]
            for col_name, dtype in cols_to_profile:
                col_profile = _profile_column(
                    conn, path, col_name, dtype, row_count, top_n
                )
                profile["columns"].append(col_profile)

        return profile
    finally:
        conn.close()
