"""Database exploration tools for LLM agents."""

from geryon.db import Database

MAX_TOP_VALUES_DISPLAY = 10
MAX_VALUES_CELL_LENGTH = 150


def list_tables(db: Database) -> str:
    """List all available tables in the database."""
    tables = db.list_tables()
    return "\n".join(f"- {table}" for table in tables)


def describe_table(db: Database, table_name: str) -> str:
    """Get schema for a specific table.

    Uses precomputed column profiles when available, falls back to
    sampling-based description otherwise.

    Parameters
    ----------
    db : Database
        Database connection
    table_name : str
        Name of table to describe

    Returns
    -------
    str
        Markdown-formatted table description
    """
    profile = db.get_profile(table_name)
    if profile is not None:
        return _describe_from_profile(profile)
    return _describe_fallback(db, table_name)


def _format_top_values(col: dict) -> str:
    """Format top values cell for a column profile."""
    top_values = col.get("top_values", [])
    if not top_values:
        return ""

    is_high = col.get("is_high_cardinality", False)
    distinct = col.get("distinct_count", 0)

    if is_high:
        display = top_values[:MAX_TOP_VALUES_DISPLAY]
    else:
        display = top_values

    parts = [f"{tv['value']} ({tv['count']:,})" for tv in display]
    cell = ", ".join(parts)

    if is_high:
        cell += f" ... [{distinct:,} distinct]"

    if len(cell) > MAX_VALUES_CELL_LENGTH:
        cell = cell[:MAX_VALUES_CELL_LENGTH] + "..."

    return cell


def _describe_from_profile(profile: dict) -> str:
    """Render table description from precomputed profile."""
    table_name = profile["table_name"]
    row_count = profile["row_count"]
    col_count = profile["column_count"]
    strategy = profile.get("profile_strategy", "standard")

    lines = [f"## Table: {table_name} ({row_count:,} rows, {col_count} columns)", ""]

    if strategy == "wide_matrix":
        gene_columns = profile.get("gene_columns", [])
        # Summarize the gene value domain from representative columns
        gene_profiles = [
            c
            for c in profile["columns"]
            if c["column_name"].upper() not in ("PATIENT_ID", "SAMPLE_ID")
        ]
        if gene_profiles:
            all_vals = set()
            for gp in gene_profiles:
                for tv in gp.get("top_values", []):
                    all_vals.add(tv["value"])
            sorted_vals = sorted(all_vals, key=lambda v: (len(v), v))
            domain_str = ", ".join(sorted_vals)
            lines.append(f"{len(gene_columns)} gene columns with values: {domain_str}")
            lines.append("")

    lines.append("| Column | Type | Distinct | Null% | Top Values |")
    lines.append("|--------|------|----------|-------|------------|")

    for col in profile["columns"]:
        null_pct = f"{col['null_rate'] * 100:.0f}%"
        top_str = _format_top_values(col)
        lines.append(
            f"| {col['column_name']} | {col['dtype']} "
            f"| {col['distinct_count']:,} | {null_pct} | {top_str} |"
        )

    if strategy == "wide_matrix":
        gene_columns = profile.get("gene_columns", [])
        if gene_columns:
            preview = ", ".join(gene_columns[:20])
            if len(gene_columns) > 20:
                preview += ", ..."
            lines.append("")
            lines.append(f"Available gene columns ({len(gene_columns)}): {preview}")

    lines.append("")
    return "\n".join(lines)


def _describe_fallback(db: Database, table_name: str) -> str:
    """Fallback: describe table from live queries (no profile available)."""
    try:
        describe_df = db.execute(f'DESCRIBE "{table_name}"')
        count_df = db.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
        row_count = int(count_df["cnt"].iloc[0])
        sample_df = db.execute(f'SELECT * FROM "{table_name}" LIMIT 5')

        lines = [f"## Table: {table_name} ({row_count:,} rows)", ""]
        lines.append("| Column | Type | Sample Values |")
        lines.append("|--------|------|---------------|")

        for idx, (_, row) in enumerate(describe_df.iterrows()):
            if idx >= 20:
                remaining = len(describe_df) - 20
                lines.append(f"| ... | ... | {remaining} more columns |")
                break

            col_name = row["column_name"]
            col_type = row["column_type"]

            col_samples = sample_df[col_name].dropna().unique()[:3]
            sample_str = ", ".join(str(v)[:40] for v in col_samples)

            lines.append(f"| {col_name} | {col_type} | {sample_str} |")

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
