"""
Compiler for translating CYCL hypothesis specs to SQL.

Provides deterministic compilation from Pydantic models to DuckDB-compatible SQL.
"""

from msk_cycl.hypothesis.spec import CohortFilter, CyclHyp, SelectCohort


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

    # Map Python operators to SQL
    sql_operators = {
        "==": "=",
        "!=": "!=",
        ">": ">",
        "<": "<",
        ">=": ">=",
        "<=": "<=",
    }

    if operator == "in":
        # Handle IN operator with list of values
        if not isinstance(value, list):
            raise ValueError(f"'in' operator requires list value, got {type(value)}")
        if len(value) == 0:
            raise ValueError("'in' operator requires non-empty list")

        escaped_values = [_escape_sql_value(v) for v in value]
        values_str = ", ".join(escaped_values)
        return f"{column} IN ({values_str})"

    # Handle other operators
    if operator not in sql_operators:
        raise ValueError(f"Unsupported operator: {operator}")

    if isinstance(value, list):
        raise ValueError(f"Operator '{operator}' does not support list values")

    sql_op = sql_operators[operator]
    escaped_value = _escape_sql_value(value)
    return f"{column} {sql_op} {escaped_value}"


def compile_select_cohort(query: SelectCohort) -> str:
    """Compile SelectCohort to SQL query."""
    if len(query.filters) == 0:
        raise ValueError("SelectCohort requires at least one filter")

    # Get table from first filter (all filters should be from same table for now)
    table = query.filters[0].table

    # Compile all filters
    where_clauses = [compile_cohort_filter(f) for f in query.filters]
    where_sql = " AND ".join(where_clauses)

    # Build query
    sql = f"SELECT * FROM {table} WHERE {where_sql}"

    return sql


def compile_hypothesis(spec: CyclHyp) -> str:
    """Compile CyclHyp specification to SQL."""
    if not isinstance(spec.query, SelectCohort):
        raise ValueError("Only select_cohort operation supported")

    return compile_select_cohort(spec.query)
