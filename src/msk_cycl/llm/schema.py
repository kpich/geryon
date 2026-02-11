"""Database schema discovery for LLM context."""

from collections.abc import Iterable

from pydantic import BaseModel, Field
from tqdm import tqdm

from msk_cycl.db import Database


class ColumnInfo(BaseModel):
    """Column metadata for LLM context."""

    name: str
    type: str
    sample_values: list[str | int | float] = Field(default_factory=list)
    distinct_count: int | None = None


class TableInfo(BaseModel):
    """Table metadata for LLM context."""

    name: str
    columns: list[ColumnInfo]
    row_count: int


class DatabaseSchema(BaseModel):
    """Complete schema information for LLM."""

    tables: list[TableInfo]


def discover_schema(
    db: Database, sample_limit: int = 10, show_progress: bool = True
) -> DatabaseSchema:
    """Extract schema information from database.

    Queries database for table names, column types, sample values, and row counts.

    Parameters
    ----------
    db : Database
        Database connection
    sample_limit : int
        Number of sample values to extract per column
    show_progress : bool
        Show progress bar during schema discovery (default: True)

    Returns
    -------
    DatabaseSchema
        Schema information for all tables
    """
    tables = []
    table_names = db.list_tables()

    table_iter: Iterable[str] = table_names
    if show_progress:
        table_iter = tqdm(table_names, desc="Discovering schema", unit="table")

    for table_name in table_iter:
        describe_df = db.execute(f'DESCRIBE "{table_name}"')
        count_df = db.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        row_count = int(count_df["cnt"].iloc[0])
        sample_df = db.execute(f'SELECT * FROM "{table_name}" LIMIT {sample_limit}')

        columns = []
        for _, row in describe_df.iterrows():
            col_name = row["column_name"]
            col_type = row["column_type"]

            col_samples = sample_df[col_name].dropna().unique()
            sample_values: list[str | int | float] = list(col_samples[:sample_limit])

            columns.append(
                ColumnInfo(
                    name=col_name,
                    type=col_type,
                    sample_values=sample_values,
                    distinct_count=None,
                )
            )

        tables.append(TableInfo(name=table_name, columns=columns, row_count=row_count))

    return DatabaseSchema(tables=tables)


def schema_to_context(schema: DatabaseSchema) -> str:
    """Format schema as markdown for LLM system prompt.

    Parameters
    ----------
    schema : DatabaseSchema
        Database schema

    Returns
    -------
    str
        Markdown-formatted schema description
    """
    lines = []

    for table in schema.tables:
        lines.append(f"## Table: {table.name} ({table.row_count:,} rows)")
        lines.append("")
        lines.append("| Column | Type | Sample Values | Distinct |")
        lines.append("|--------|------|---------------|----------|")

        for col in table.columns:
            sample_str = ""
            if col.sample_values:
                # Format sample values nicely
                samples = [str(v) for v in col.sample_values[:5]]
                sample_str = ", ".join(f'"{s}"' if " " in s else s for s in samples)
                if len(col.sample_values) > 5:
                    sample_str += ", ..."

            distinct_str = str(col.distinct_count) if col.distinct_count else ""

            lines.append(f"| {col.name} | {col.type} | {sample_str} | {distinct_str} |")

        lines.append("")

    return "\n".join(lines)
