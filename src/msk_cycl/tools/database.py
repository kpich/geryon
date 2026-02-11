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
        summarize_df = db.execute(f'SUMMARIZE "{table_name}"')
        row_count = int(summarize_df["count"].max())

        lines = [f"## Table: {table_name} ({row_count:,} rows)", ""]
        lines.append("| Column | Type | Min | Max | Approx Distinct |")
        lines.append("|--------|------|-----|-----|-----------------|")

        for idx, (_, row) in enumerate(summarize_df.iterrows()):
            if idx >= 20:
                remaining = len(summarize_df) - 20
                lines.append(f"| ... | ... | ... | ... | {remaining} more columns |")
                break

            col_name = row["column_name"]
            col_type = row["column_type"]
            min_val = str(row.get("min", "")) if row.get("min") is not None else ""
            max_val = str(row.get("max", "")) if row.get("max") is not None else ""
            approx_unique = row.get("approx_unique", "")

            if len(min_val) > 40:
                min_val = min_val[:37] + "..."
            if len(max_val) > 40:
                max_val = max_val[:37] + "..."

            lines.append(
                f"| {col_name} | {col_type} | {min_val} | {max_val}"
                f" | {approx_unique} |"
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
