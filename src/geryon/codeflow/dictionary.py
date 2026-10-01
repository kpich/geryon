"""A data dictionary the agents write themselves.

The ETL drops cBioPortal's column descriptions, so every agent otherwise re-derives what
the data means (when ``OS_MONTHS`` starts, how CNA joins), and they have disagreed. An
entry says how to read the data, as a codebook would; what the data shows about patients
is a finding and stays out. An entry is added only with a check script that asserts it
and passes in the sandbox, so the gate is executable evidence rather than the model's
confidence. The dictionary is shown to the generator, critic and narrator of every later
session that reads the same data version.

The store lives next to the sessions (``<output_dir>/data_dictionary.jsonl``), not with
the data: it is a record of what runs have learned, and it is backed up with them.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from geryon.codeflow.agent_tools import format_run, run_in_sandbox
from geryon.etl.data_version import resolve_data_version
from geryon.sandbox import SandboxLimits
from geryon.workflow.session import SessionConfig

DICTIONARY_FILENAME = "data_dictionary.jsonl"

Recorder = Literal["generator", "critic", "seed"]


class DictionaryEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entry: str
    check_code: str = Field(description="Script whose asserts establish the entry")
    data_version: str
    session_id: str
    recorded_by: Recorder
    supersedes: str | None = Field(
        default=None, description="entry_id of an earlier entry this one corrects"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def short_id(self) -> str:
        return self.entry_id[:8]


def has_assert(code: str) -> bool:
    """Whether a script contains an ``assert`` statement (False if it won't parse)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


class DataDictionary:
    """Append-only JSONL of entries, shared by every session under one output dir."""

    def __init__(self, output_dir: Path | str):
        self.path = Path(output_dir) / DICTIONARY_FILENAME

    def load_all(self) -> list[DictionaryEntry]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [
                DictionaryEntry.model_validate_json(line) for line in f if line.strip()
            ]

    def current(self, data_version: str) -> list[DictionaryEntry]:
        """Entries recorded on this data version that no later entry has corrected."""
        entries = self.load_all()
        superseded = {e.supersedes for e in entries if e.supersedes}
        return [
            f
            for f in entries
            if f.data_version == data_version and f.entry_id not in superseded
        ]

    def find(self, entry_id: str) -> DictionaryEntry | None:
        short = entry_id[:8]
        for f in self.load_all():
            if f.entry_id == entry_id or f.short_id() == short:
                return f
        return None

    def append(self, entry: DictionaryEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(entry.model_dump_json() + "\n")


def format_entries(entries: list[DictionaryEntry]) -> str | None:
    """Render entries for a system prompt; None when there are none."""
    if not entries:
        return None
    return "\n".join(f"- [{e.short_id()}] {e.entry}" for e in entries)


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
