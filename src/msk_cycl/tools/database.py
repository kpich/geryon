"""Database exploration tools for LLM agents."""

from msk_cycl.db import Database


def list_tables(db: Database) -> str:
    """List all available tables in the database."""
    tables = db.list_tables()
    return "\n".join(f"- {table}" for table in tables)


def describe_table(db: Database, table_name: str) -> str:
    """Get schema for a specific table (columns, types, sample values).

    Parameters
    ----------
    db : Database
        Database connection
    table_name : str
        Name of table to describe

    Returns
    -------
    str
        Markdown-formatted table schema with columns, types, and samples
    """
    try:
        # Get column information
        describe_sql = f'DESCRIBE "{table_name}"'
        describe_df = db.execute(describe_sql)

        # Get row count
        count_sql = f'SELECT COUNT(*) as cnt FROM "{table_name}"'
        count_df = db.execute(count_sql)
        row_count = int(count_df["cnt"].iloc[0])

        lines = [f"## Table: {table_name} ({row_count:,} rows)", ""]
        lines.append("| Column | Type | Sample Values | Distinct |")
        lines.append("|--------|------|---------------|----------|")

        # Limit to 20 columns to avoid overwhelming output
        for idx, (_, row) in enumerate(describe_df.iterrows()):
            if idx >= 20:
                remaining = len(describe_df) - 20
                lines.append(f"| ... | ... | ... | {remaining} more columns |")
                break

            col_name = row["column_name"]
            col_type = row["column_type"]

            # Get sample values (up to 3)
            sample_values = []
            distinct_count = ""

            try:
                if "VARCHAR" in col_type or "TEXT" in col_type:
                    sample_sql = (
                        f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL LIMIT 3'
                    )
                    sample_df = db.execute(sample_sql)
                    sample_values = sample_df[col_name].tolist()

                    count_sql = (
                        f'SELECT COUNT(DISTINCT "{col_name}") as cnt '
                        f'FROM "{table_name}"'
                    )
                    count_df = db.execute(count_sql)
                    distinct_count = str(int(count_df["cnt"].iloc[0]))
                elif (
                    "DOUBLE" in col_type
                    or "INTEGER" in col_type
                    or "BIGINT" in col_type
                ):
                    sample_sql = (
                        f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                        f'WHERE "{col_name}" IS NOT NULL LIMIT 3'
                    )
                    sample_df = db.execute(sample_sql)
                    sample_values = sample_df[col_name].tolist()

            except Exception:
                # Skip sample values if query fails
                pass

            sample_str = ", ".join(str(v) for v in sample_values)
            lines.append(
                f"| {col_name} | {col_type} | {sample_str} | {distinct_count} |"
            )

        lines.append("")
        return "\n".join(lines)

    except Exception as e:
        return (
            f"ERROR: Could not describe table '{table_name}': "
            f"{type(e).__name__}: {str(e)}"
        )


def query_data(db: Database, sql: str) -> str:
    """Run SQL query (SELECT only, max 100 rows).

    Parameters
    ----------
    db : Database
        Database connection
    sql : str
        SQL query to execute (must be SELECT)

    Returns
    -------
    str
        Markdown-formatted query results
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "ERROR: Only SELECT queries allowed"

    # Add LIMIT if not present
    if "LIMIT" not in sql.upper():
        sql = f"{sql} LIMIT 100"

    try:
        df = db.execute(sql)
        return df.to_string(index=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"
