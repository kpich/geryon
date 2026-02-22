"""
DuckDB database connection and query execution.

Provides in-memory database with auto-registered parquet files.
"""

import json
from pathlib import Path
import threading

import duckdb
import pandas as pd  # type: ignore


def _get_table_name(parquet_path: Path) -> str:
    """Extract table name from parquet filename."""
    stem = parquet_path.stem

    if stem.startswith("data_"):
        return stem[5:]

    return stem


class Database:
    """DuckDB connection manager for querying parquet files."""

    def __init__(self, parquet_dir: str | Path):
        """
        Initialize DuckDB connection and register parquet files as tables.

        Parameters
        ----------
        parquet_dir : str | Path
            Directory containing parquet files from ETL output
        """
        self.parquet_dir = Path(parquet_dir)

        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.parquet_dir}")

        if not self.parquet_dir.is_dir():
            raise ValueError(f"Not a directory: {self.parquet_dir}")

        self.conn = duckdb.connect(":memory:")
        self._lock = threading.Lock()
        self.profiles: dict[str, dict] = {}

        self._register_tables()
        self._load_profiles()

    def _register_tables(self) -> None:
        """Scan parquet directory and register each file as a table."""
        parquet_files = list(self.parquet_dir.glob("*.parquet"))

        if len(parquet_files) == 0:
            raise ValueError(f"No parquet files found in {self.parquet_dir}")

        for parquet_file in parquet_files:
            table_name = _get_table_name(parquet_file)

            # Create view pointing to parquet file
            # Using views is more efficient than loading into memory
            # Quote table name to handle special characters like hyphens
            sql = f"CREATE VIEW \"{table_name}\" AS SELECT * FROM '{parquet_file}'"
            self.conn.execute(sql)

    def _load_profiles(self) -> None:
        """Load column profile JSON files alongside parquet."""
        for profile_path in self.parquet_dir.glob("*.profile.json"):
            try:
                profile = json.loads(profile_path.read_text())
                table_name = profile.get("table_name", profile_path.stem)
                self.profiles[table_name] = profile
            except (json.JSONDecodeError, OSError):
                continue

    def get_profile(self, table_name: str) -> dict | None:
        """Look up precomputed column profile for a table."""
        return self.profiles.get(table_name)

    def create_view(self, name: str, sql: str) -> None:
        """Register an in-session DuckDB view."""
        with self._lock:
            self.conn.execute(f'CREATE OR REPLACE VIEW "{name}" AS ({sql})')

    def execute(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to execute

        Returns
        -------
        pd.DataFrame
            Query results
        """
        with self._lock:
            result = self.conn.execute(sql)
            return result.df()

    def list_tables(self) -> list[str]:
        """List all registered tables."""
        result = self.conn.execute("SHOW TABLES")
        return [row[0] for row in result.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

    def __enter__(self) -> "Database":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.close()
