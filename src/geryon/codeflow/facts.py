"""Verified facts about the data, recorded by the agents themselves.

Every agent otherwise re-derives what the data means (when ``OS_MONTHS`` starts, how
CNA joins), and they have disagreed. A fact is recorded only with a check script that
asserts it and passes in the sandbox, so the gate is executable evidence rather than
the model's confidence. Facts are shown to the generator, critic and narrator of every
later session that reads the same data version.

The store lives next to the sessions (``<output_dir>/data_facts.jsonl``), not with the
data: it is a record of what runs have learned, and it is backed up with them.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, Field

FACTS_FILENAME = "data_facts.jsonl"

Recorder = Literal["generator", "critic", "seed"]


class DataFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact: str
    check_code: str = Field(description="Script whose asserts establish the fact")
    data_version: str
    session_id: str
    recorded_by: Recorder
    supersedes: str | None = Field(
        default=None, description="fact_id of an earlier fact this one corrects"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def short_id(self) -> str:
        return self.fact_id[:8]


def has_assert(code: str) -> bool:
    """Whether a script contains an ``assert`` statement (False if it won't parse)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


class DataFactStore:
    """Append-only JSONL of facts, shared by every session under one output dir."""

    def __init__(self, output_dir: Path | str):
        self.path = Path(output_dir) / FACTS_FILENAME

    def load_all(self) -> list[DataFact]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [DataFact.model_validate_json(line) for line in f if line.strip()]

    def current(self, data_version: str) -> list[DataFact]:
        """Facts recorded on this data version that no later fact has corrected."""
        facts = self.load_all()
        superseded = {f.supersedes for f in facts if f.supersedes}
        return [
            f
            for f in facts
            if f.data_version == data_version and f.fact_id not in superseded
        ]

    def find(self, fact_id: str) -> DataFact | None:
        short = fact_id[:8]
        for f in self.load_all():
            if f.fact_id == fact_id or f.short_id() == short:
                return f
        return None

    def append(self, fact: DataFact) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(fact.model_dump_json() + "\n")


def format_facts(facts: list[DataFact]) -> str | None:
    """Render facts for a system prompt; None when there are none."""
    if not facts:
        return None
    return "\n".join(f"- [{f.short_id()}] {f.fact}" for f in facts)
