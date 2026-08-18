"""JSONL storage for code hypotheses.

Mirrors the legacy ``HypothesisStore`` format (metadata line 1, one record per line,
atomic rewrite) but for ``CodeHypothesis`` and with no spec/version coupling.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from geryon.codeflow.chains import DEFAULT_CHAIN
from geryon.codeflow.models import CodeHypothesis

HYPOTHESES_FILENAME = "hypotheses.jsonl"


class _Metadata(BaseModel):
    record_type: Literal["metadata"] = "metadata"
    session_id: str
    created_at: datetime
    format: Literal["codeflow"] = "codeflow"
    # Which line of investigation this session extends. Defaulted so sessions written
    # before chains existed load as the main chain.
    chain: str = DEFAULT_CHAIN


class _Record(BaseModel):
    record_type: Literal["hypothesis"] = "hypothesis"
    data: CodeHypothesis
    written_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CodeHypothesisStore:
    """Append-only JSONL store; the directory is the session."""

    def __init__(self, storage_dir: Path | str, chain: str = DEFAULT_CHAIN):
        self.storage_dir = Path(storage_dir)
        self.chain = chain
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _path(self) -> Path:
        return self.storage_dir / HYPOTHESES_FILENAME

    def save(self, hypothesis: CodeHypothesis) -> None:
        """Append a hypothesis, writing the metadata header if the file is new."""
        if not self._path.exists():
            meta = _Metadata(
                session_id=hypothesis.session_id,
                created_at=datetime.now(UTC),
                chain=self.chain,
            )
            self._path.write_text(meta.model_dump_json() + "\n")

        with open(self._path, "a") as f:
            f.write(_Record(data=hypothesis).model_dump_json() + "\n")

    def save_all(self, hypotheses: list[CodeHypothesis]) -> None:
        """Rewrite the file with the given hypotheses (atomic temp+rename)."""
        if not hypotheses:
            return
        tmp = self._path.with_suffix(".jsonl.tmp")
        meta = _Metadata(
            session_id=hypotheses[0].session_id,
            created_at=datetime.now(UTC),
            chain=self.chain,
        )
        with open(tmp, "w") as f:
            f.write(meta.model_dump_json() + "\n")
            for hyp in hypotheses:
                f.write(_Record(data=hyp).model_dump_json() + "\n")
        tmp.rename(self._path)

    def load(self) -> list[CodeHypothesis]:
        """Load all hypotheses (skipping the metadata header)."""
        if not self._path.exists():
            return []
        out: list[CodeHypothesis] = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("record_type") == "metadata":
                    continue
                out.append(_Record(**data).data)
        return out
