"""Tests for database exploration tools."""

from unittest.mock import Mock

import pandas as pd

from geryon.tools.database import describe_table, list_tables, query_data


def test_list_tables():
    """Test list_tables formats table names as markdown list."""
    mock_db = Mock()
    mock_db.list_tables.return_value = ["clinical_patient", "CNA", "mutations"]

    result = list_tables(mock_db)

    assert "- clinical_patient" in result
    assert "- CNA" in result
    assert "- mutations" in result


def test_describe_table_basic():
    """Test describe_table formats table schema."""
    mock_db = Mock()
    mock_db.get_profile.return_value = None

    # Mock DESCRIBE query
    describe_df = pd.DataFrame(
        {
            "column_name": ["PATIENT_ID", "CANCER_TYPE", "OS_MONTHS"],
            "column_type": ["VARCHAR", "VARCHAR", "DOUBLE"],
        }
    )
    # Mock COUNT query
    count_df = pd.DataFrame({"cnt": [100]})

    sample_df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002"],
            "CANCER_TYPE": ["Breast", "Lung"],
            "OS_MONTHS": [12.5, 8.3],
        }
    )

    mock_db.execute.side_effect = [describe_df, count_df, sample_df]

    result = describe_table(mock_db, "clinical_patient")

    assert "## Table: clinical_patient (100 rows)" in result
    assert "PATIENT_ID" in result
    assert "VARCHAR" in result
    assert "OS_MONTHS" in result
    assert "DOUBLE" in result


def test_describe_table_shows_sample_values():
    """Test describe_table includes sample values for VARCHAR columns."""
    mock_db = Mock()
    mock_db.get_profile.return_value = None

    # Mock DESCRIBE query
    describe_df = pd.DataFrame(
        {
            "column_name": ["PATIENT_ID", "CANCER_TYPE"],
            "column_type": ["VARCHAR", "VARCHAR"],
        }
    )
    # Mock COUNT query
    count_df = pd.DataFrame({"cnt": [50]})
    sample_df = pd.DataFrame(
        {
            "PATIENT_ID": ["P001", "P002", "P003"],
            "CANCER_TYPE": ["Breast", "Lung", "Colon"],
        }
    )

    mock_db.execute.side_effect = [describe_df, count_df, sample_df]

    result = describe_table(mock_db, "clinical_patient")

    assert "P001" in result or "P002" in result


def test_describe_table_limits_to_20_columns():
    """Test describe_table limits output to 20 columns."""
    mock_db = Mock()
    mock_db.get_profile.return_value = None

    # Create table with 30 columns
    columns = [f"COL_{i}" for i in range(30)]
    types = ["VARCHAR"] * 30

    describe_df = pd.DataFrame({"column_name": columns, "column_type": types})
    count_df = pd.DataFrame({"cnt": [100]})

    sample_df = pd.DataFrame({f"COL_{i}": ["val"] for i in range(30)})

    mock_db.execute.side_effect = [describe_df, count_df, sample_df]

    result = describe_table(mock_db, "wide_table")

    # Should show message about remaining columns
    assert "10 more columns" in result


def test_describe_table_handles_error():
    """Test describe_table returns error message on failure."""
    mock_db = Mock()
    mock_db.get_profile.return_value = None
    mock_db.execute.side_effect = Exception("Table not found")

    result = describe_table(mock_db, "nonexistent")

    assert "ERROR" in result
    assert "nonexistent" in result


def test_query_data_executes_select():
    """Test query_data executes SELECT queries."""
    mock_db = Mock()
    result_df = pd.DataFrame({"PATIENT_ID": ["P001", "P002"], "AGE": [65, 52]})
    mock_db.execute.return_value = result_df

    result = query_data(mock_db, "SELECT * FROM clinical_patient")

    assert "PATIENT_ID" in result
    assert "P001" in result
    mock_db.execute.assert_called_once()


def test_query_data_adds_limit_if_missing():
    """Test query_data adds LIMIT 100 if not present."""
    mock_db = Mock()
    mock_db.execute.return_value = pd.DataFrame({"col": [1]})

    query_data(mock_db, "SELECT * FROM foo")

    # Check that LIMIT was added
    call_args = mock_db.execute.call_args[0][0]
    assert "LIMIT 100" in call_args


def test_query_data_preserves_existing_limit():
    """Test query_data respects user-provided LIMIT."""
    mock_db = Mock()
    mock_db.execute.return_value = pd.DataFrame({"col": [1]})

    query_data(mock_db, "SELECT * FROM foo LIMIT 10")

    # Should not add another LIMIT
    call_args = mock_db.execute.call_args[0][0]
    assert call_args.count("LIMIT") == 1


def test_query_data_rejects_non_select():
    """Test query_data rejects non-SELECT queries."""
    mock_db = Mock()

    dangerous_queries = [
        "DROP TABLE foo",
        "DELETE FROM foo",
        "UPDATE foo SET bar = 1",
        "INSERT INTO foo VALUES (1)",
        "drop table foo",  # lowercase
        "update foo set bar = 1",  # lowercase
    ]

    for query in dangerous_queries:
        result = query_data(mock_db, query)
        assert "ERROR" in result
        assert "Only SELECT queries allowed" in result
        mock_db.execute.assert_not_called()
        mock_db.reset_mock()


def test_query_data_handles_sql_error():
    """Test query_data returns error message on SQL failure."""
    mock_db = Mock()
    mock_db.execute.side_effect = Exception("Column not found")

    result = query_data(mock_db, "SELECT bad_column FROM foo")

    assert "ERROR" in result
    assert "Exception" in result


def test_describe_table_uses_profile():
    """When profile exists, describe_table uses profile-based rendering."""
    mock_db = Mock()
    mock_db.get_profile.return_value = {
        "table_name": "clinical_patient",
        "row_count": 1000,
        "column_count": 3,
        "profile_strategy": "standard",
        "columns": [
            {
                "column_name": "PATIENT_ID",
                "dtype": "VARCHAR",
                "null_count": 0,
                "null_rate": 0.0,
                "distinct_count": 1000,
                "total_count": 1000,
                "top_values": [{"value": "P001", "count": 1}],
                "is_high_cardinality": True,
            },
            {
                "column_name": "SEX",
                "dtype": "VARCHAR",
                "null_count": 0,
                "null_rate": 0.0,
                "distinct_count": 2,
                "total_count": 1000,
                "top_values": [
                    {"value": "Female", "count": 520},
                    {"value": "Male", "count": 480},
                ],
                "is_high_cardinality": False,
            },
        ],
    }

    result = describe_table(mock_db, "clinical_patient")

    assert "1,000 rows" in result
    assert "3 columns" in result
    assert "Distinct" in result
    assert "Null%" in result
    assert "Female (520)" in result
    assert "Male (480)" in result
    # Should NOT have called execute (no fallback)
    mock_db.execute.assert_not_called()


def test_describe_table_fallback_without_profile():
    """No profile → falls back to old sample-based behavior."""
    mock_db = Mock()
    mock_db.get_profile.return_value = None

    describe_df = pd.DataFrame(
        {
            "column_name": ["PATIENT_ID", "AGE"],
            "column_type": ["VARCHAR", "BIGINT"],
        }
    )
    count_df = pd.DataFrame({"cnt": [50]})
    sample_df = pd.DataFrame({"PATIENT_ID": ["P001", "P002"], "AGE": [65, 52]})
    mock_db.execute.side_effect = [describe_df, count_df, sample_df]

    result = describe_table(mock_db, "clinical_patient")

    assert "Sample Values" in result
    assert "P001" in result or "P002" in result
