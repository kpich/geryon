"""LLM factory, exploration tools and sandbox runner shared by the generator
(agent.py) and the critic (critic.py)."""

from __future__ import annotations

import json
from typing import NamedTuple

from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from geryon.codeflow.dictionary import (
    DataDictionary,
    DictionaryEntry,
    Recorder,
    format_entries,
    has_assert,
)
from geryon.db import Database
from geryon.etl.data_version import resolve_data_version
from geryon.llm.providers.base import LLMResponse
from geryon.sandbox import SandboxLimits, ScriptRun, run_script
from geryon.tools.database import describe_table, list_tables, query_data
from geryon.workflow.session import SessionConfig

# How much stdout/stderr to feed back to the agent after a run.
_OUTPUT_TAIL_CHARS = 6000


class MessageUsage(NamedTuple):
    """Token usage summed over one LLM phase (generation, critic, narration)."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    n_llm_calls: int


def sum_message_usage(messages: list) -> MessageUsage:
    """Sum token usage across a LangGraph/LangChain message list.

    Reads each message's ``usage_metadata`` (set on AI responses). On Bedrock,
    ``input_tokens`` is the uncached input only and the ``input_token_details`` cache
    counts are separate from it, the same disjoint buckets the cost plot prices.
    """
    inp = out = tot = cache_read = cache_create = calls = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        i = int(usage.get("input_tokens", 0) or 0)
        o = int(usage.get("output_tokens", 0) or 0)
        inp += i
        out += o
        tot += int(usage.get("total_tokens", 0) or 0) or (i + o)
        details = usage.get("input_token_details") or {}
        cache_read += int(details.get("cache_read", 0) or 0)
        cache_create += int(details.get("cache_creation", 0) or 0)
        calls += 1
    return MessageUsage(inp, out, tot, cache_read, cache_create, calls)


def usage_from_response(response: LLMResponse) -> MessageUsage:
    """Map one provider ``LLMResponse.usage`` dict into a ``MessageUsage``.

    ``prompt_tokens`` is the uncached input; ``cache_read_tokens`` and
    ``cache_write_tokens`` are separate from it, as the cost plot expects.
    """
    usage = response.usage or {}
    inp = int(usage.get("prompt_tokens", 0) or 0)
    out = int(usage.get("completion_tokens", 0) or 0)
    tot = int(usage.get("total_tokens", 0) or 0) or (inp + out)
    return MessageUsage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=tot,
        cache_read_tokens=int(usage.get("cache_read_tokens", 0) or 0),
        cache_creation_tokens=int(usage.get("cache_write_tokens", 0) or 0),
        n_llm_calls=1,
    )


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


def session_data_version(config: SessionConfig) -> str:
    return config.data_version or resolve_data_version(config.parquet_dir)


def dictionary_store(config: SessionConfig) -> DataDictionary:
    # Without an output dir (tests, ad-hoc runs) the dictionary stays with the session.
    return DataDictionary(config.output_dir or config.storage_dir)


def data_dictionary_text(config: SessionConfig) -> str | None:
    """The data dictionary for this session's data version, rendered for a prompt."""
    if not config.include_data_dictionary:
        return None
    store = dictionary_store(config)
    return format_entries(store.current(session_data_version(config)))


def make_dictionary_tool(
    config: SessionConfig, limits: SandboxLimits, recorded_by: Recorder
):
    """A tool that adds a dictionary entry, once a script asserting it passes."""
    store = dictionary_store(config)

    @tool
    def add_to_data_dictionary(
        entry: str, check_code: str, supersedes: str | None = None
    ) -> str:
        """Add an entry to the data dictionary that every later session sees.

        The ETL dropped the source's column descriptions, so this dictionary is
        what agents have instead. Write an entry the way a codebook would: what a
        table, column or value means, its units or coding, how tables join, what
        time zero is, coverage or missingness, a trap that makes a naive query
        wrong. It must hold whatever cancer type or question someone is studying.

        What the data shows about patients is a finding, not an entry: rates or
        counts within a cohort, relationships between variables ("X-mutant tumors
        have higher Y"), effect sizes, or an interpretation ("consistent with a
        non-secretor phenotype"). Findings go in hypotheses and critiques, where
        they can be challenged; stored here, later agents would take them as
        settled. If an entry only holds for one cancer type, it is a finding.

        `check_code` must contain `assert` statements that establish the entry and
        must run cleanly in the sandbox (`from geryon_runtime import db`). Set
        `supersedes` to the id of an earlier entry this one corrects.
        """
        print("[TOOL] add_to_data_dictionary called")
        if not has_assert(check_code):
            return (
                "✗ NOT SAVED: check_code has no assert statement (or doesn't parse). "
                "Assert the entry against the data."
            )
        old = None
        if supersedes:
            old = store.find(supersedes)
            if old is None:
                return f"✗ NOT SAVED: no entry with id '{supersedes}' to supersede."
        run = run_in_sandbox(config, check_code, limits)
        if not run.success:
            return "✗ NOT SAVED: the check failed.\n" + format_run(run)
        saved = DictionaryEntry(
            entry=entry,
            check_code=check_code,
            data_version=session_data_version(config),
            session_id=config.session_id,
            recorded_by=recorded_by,
            supersedes=old.entry_id if old else None,
        )
        store.append(saved)
        return f"✓ Entry saved [{saved.short_id()}]."

    return add_to_data_dictionary


def build_chat_model(config: SessionConfig):
    """Build a LangChain chat model from the session's provider config."""
    if config.provider_type == "openai":
        kwargs: dict = {
            "model": config.model,
            "temperature": 0.8,
            "max_tokens": 16384,
        }
        if config.base_url:
            kwargs["openai_api_base"] = config.base_url
        if config.api_key:
            kwargs["openai_api_key"] = config.api_key
        return ChatOpenAI(**kwargs)  # type: ignore[arg-type]
    elif config.provider_type == "aws_bedrock":
        boto_config = BotoConfig(
            read_timeout=300, connect_timeout=30, retries={"max_attempts": 2}
        )
        kwargs = {
            "model_id": config.model,
            # Current Claude models 400 on temperature/top_p/top_k, so none are sent.
            "model_kwargs": {"max_tokens": 16384},
            "config": boto_config,
        }
        if config.aws_region:
            kwargs["region_name"] = config.aws_region
        if config.aws_profile:
            kwargs["credentials_profile_name"] = config.aws_profile
        if "arn:" in config.model or "anthropic" in config.model.lower():
            kwargs["provider"] = "anthropic"
        return ChatBedrock(**kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown provider type: {config.provider_type}")
