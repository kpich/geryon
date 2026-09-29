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

from pydantic import BaseModel, Field

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
