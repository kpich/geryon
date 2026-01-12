"""Database schema discovery for LLM context."""

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

    # Wrap table iteration with tqdm if requested
    table_iter = table_names
    if show_progress:
        table_iter = tqdm(table_names, desc="Discovering schema", unit="table")

    for table_name in table_iter:
        # Get column info using DESCRIBE
        # Quote table name to handle special characters
        describe_sql = f'DESCRIBE "{table_name}"'
        describe_df = db.execute(describe_sql)

        columns = []

        # Show progress for columns too
        column_iter = describe_df.iterrows()
        if show_progress:
            column_iter = tqdm(
                describe_df.iterrows(),
                desc=f"  {table_name}",
                total=len(describe_df),
                unit="col",
                leave=False,
            )

        for _, row in column_iter:
            col_name = row["column_name"]
            col_type = row["column_type"]

            # Get sample values for categorical-looking columns
            sample_values: list[str | int | float] = []
            distinct_count = None

            # Only sample for string columns or small numeric columns
            if "VARCHAR" in col_type or "TEXT" in col_type:
                # Quote identifiers to handle special characters
                sample_sql = (
                    f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                    f'WHERE "{col_name}" IS NOT NULL LIMIT {sample_limit}'
                )
                try:
                    sample_df = db.execute(sample_sql)
                    sample_values = sample_df[col_name].tolist()

                    # Get distinct count
                    count_sql = (
                        f'SELECT COUNT(DISTINCT "{col_name}") as cnt '
                        f'FROM "{table_name}"'
                    )
                    count_df = db.execute(count_sql)
                    distinct_count = int(count_df["cnt"].iloc[0])
                except Exception:
                    # If sampling fails, just skip
                    pass

            columns.append(
                ColumnInfo(
                    name=col_name,
                    type=col_type,
                    sample_values=sample_values,
                    distinct_count=distinct_count,
                )
            )

        # Get row count
        count_sql = f'SELECT COUNT(*) as cnt FROM "{table_name}"'
        count_df = db.execute(count_sql)
        row_count = int(count_df["cnt"].iloc[0])

        tables.append(
            TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
            )
        )

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
