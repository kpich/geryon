"""
Compiler for translating Geryon hypothesis specs to SQL.

Provides deterministic compilation from Pydantic models to DuckDB-compatible SQL.
"""

from geryon.legacy.lang.spec import CohortFilter, SelectCohort


def _escape_sql_value(value: str | int | float) -> str:
    """Escape value for SQL (basic protection, DuckDB params are better)."""
    if isinstance(value, str):
        # Escape single quotes by doubling them
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return str(value)


def compile_cohort_filter(filter: CohortFilter) -> str:
    """Compile single filter to SQL WHERE clause fragment."""
    column = filter.column
    operator = filter.operator
    value = filter.value

    sql_operators = {
        "==": "=",
        "!=": "!=",
        ">": ">",
        "<": "<",
        ">=": ">=",
        "<=": "<=",
    }

    if operator == "in":
        if not isinstance(value, list):
            raise ValueError(f"'in' operator requires list value, got {type(value)}")
        if len(value) == 0:
            raise ValueError("'in' operator requires non-empty list")

        escaped_values = [_escape_sql_value(v) for v in value]
        values_str = ", ".join(escaped_values)
        return f'"{column}" IN ({values_str})'

    if operator not in sql_operators:
        raise ValueError(f"Unsupported operator: {operator}")

    if isinstance(value, list):
        raise ValueError(f"Operator '{operator}' does not support list values")

    sql_op = sql_operators[operator]
    escaped_value = _escape_sql_value(value)
    # Quote column name to handle special characters
    return f'"{column}" {sql_op} {escaped_value}'


def compile_select_cohort(query: SelectCohort) -> str:
    """Compile SelectCohort to SQL query."""
    if len(query.filters) == 0:
        raise ValueError("SelectCohort requires at least one filter")

    table = query.filters[0].table

    where_clauses = [compile_cohort_filter(f) for f in query.filters]
    where_sql = " AND ".join(where_clauses)

    sql = f'SELECT * FROM "{table}" WHERE {where_sql}'

    return sql


def compile_select_cohort_ids(query: SelectCohort) -> str:
    """
    Compile SelectCohort to SQL query that returns PATIENT_ID only.

    INTERNAL USE: This is used by the executor to get cohort IDs.
    Returns SQL like: "SELECT PATIENT_ID FROM table WHERE ..."
    """
    if len(query.filters) == 0:
        raise ValueError("SelectCohort requires at least one filter")

    table = query.filters[0].table
    where_clauses = [compile_cohort_filter(f) for f in query.filters]
    where_sql = " AND ".join(where_clauses)

    # Quote identifiers to handle special characters
    sql = f'SELECT "PATIENT_ID" FROM "{table}" WHERE {where_sql}'
    return sql


def compile_select_cohort_ids_via_join(
    query: SelectCohort,
    sample_key_column: str,
    join_table: str,
    join_column: str,
) -> str:
    """Compile SelectCohort to SQL that resolves PATIENT_ID through a JOIN.

    Used for tables (e.g. mutations_extended) that don't have PATIENT_ID
    directly but can be joined to a table that does (e.g. clinical_sample).
    """
    if len(query.filters) == 0:
        raise ValueError("SelectCohort requires at least one filter")

    table = query.filters[0].table
    where_clauses = [compile_cohort_filter(f) for f in query.filters]
    where_sql = " AND ".join(where_clauses)

    sql = (
        f'SELECT DISTINCT "{join_table}"."PATIENT_ID"'
        f' FROM "{table}"'
        f' JOIN "{join_table}"'
        f' ON "{table}"."{sample_key_column}" = "{join_table}"."{join_column}"'
        f" WHERE {where_sql}"
    )
    return sql
