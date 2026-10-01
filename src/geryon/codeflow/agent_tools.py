"""LangChain tools the generator and critic both use: read-only data exploration and
running a script in the sandbox."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from geryon.db import Database
from geryon.sandbox import SandboxLimits, ScriptRun, run_script
from geryon.tools.database import describe_table, list_tables, query_data
from geryon.workflow.session import SessionConfig

# How much stdout/stderr to feed back to the agent after a run.
_OUTPUT_TAIL_CHARS = 6000


def _tail(text: str, limit: int = _OUTPUT_TAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return "...(truncated)...\n" + text[-limit:]


def format_run(run: ScriptRun) -> str:
    """Render a ScriptRun for an LLM agent to read."""
    if run.timed_out:
        status = "TIMEOUT"
    elif run.success:
        status = "OK"
    else:
        status = f"EXIT {run.exit_code}"

    lines = [f"status: {status} ({run.duration_seconds:.1f}s)"]
    if run.error:
        lines.append(f"sandbox error: {run.error}")
    if run.result is not None:
        lines.append(
            "reported result: " + json.dumps(run.result.model_dump(), default=str)
        )
    else:
        lines.append("reported result: (none — script did not call report())")
    if run.stdout.strip():
        lines.append("stdout:\n" + _tail(run.stdout))
    if run.stderr.strip():
        lines.append("stderr:\n" + _tail(run.stderr))
    return "\n".join(lines)


def run_in_sandbox(
    config: SessionConfig, code: str, limits: SandboxLimits
) -> ScriptRun:
    return run_script(
        code,
        config.parquet_dir,
        timeout=config.sandbox_timeout_seconds,
        limits=limits,
    )


def make_explore_tools(db: Database) -> list:
    """Read-only data-exploration tools (shared by agent and critic)."""

    @tool
    def list_tables_tool() -> str:
        """List all available tables in the database."""
        return list_tables(db)

    @tool
    def describe_table_tool(table_name: str) -> str:
        """Get schema (columns, types, sample values) for a specific table."""
        return describe_table(db, table_name)

    @tool
    def query_data_tool(sql: str) -> str:
        """Run a read-only SELECT (or WITH ... SELECT) query, max 100 rows."""
        return query_data(db, sql)

    return [list_tables_tool, describe_table_tool, query_data_tool]


def make_run_python_tool(config: SessionConfig, limits: SandboxLimits):
    """A run_python tool that executes a script in the sandbox (stores nothing)."""

    @tool
    def run_python(code: str) -> str:
        """Execute a Python script in the sandbox and return its output.

        The script may `from geryon_runtime import db, report`. Use this to test an
        analysis or probe a suspicion. Nothing is stored.
        """
        print("[TOOL] run_python called")
        return format_run(run_in_sandbox(config, code, limits))

    return run_python
