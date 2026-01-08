"""JSONL file format schema with versioning.

This module defines the structure of JSONL session files for storing labeled
hypotheses. Each file starts with a metadata record, followed by hypothesis
records.

File structure:
    Line 1: SessionFileMetadata record
    Line 2+: HypothesisRecord records

Each line is a complete JSON object with versioning for future migrations.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from msk_cycl.labeling.models import LabeledHypothesis


class SessionFileMetadata(BaseModel):
    """Metadata record for JSONL session file (always first line)."""

    file_format_version: int = Field(
        default=1, description="Version for future format migrations"
    )
    record_type: Literal["metadata"] = "metadata"

    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")
    cycl_version: str = Field(default="0.1.0", description="CYCL software version")


class HypothesisRecord(BaseModel):
    """Hypothesis record in JSONL file (one per line after metadata)."""

    file_format_version: int = Field(
        default=1, description="Version for future format migrations"
    )
    record_type: Literal["hypothesis"] = "hypothesis"

    data: LabeledHypothesis = Field(..., description="The labeled hypothesis")
    written_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was written",
    )


class SessionFile(BaseModel):
    """Complete session file schema (for validation, not direct storage).

    This model represents the logical structure of a session file but is not
    used for actual file I/O. The storage layer writes records line-by-line.
    """

    metadata: SessionFileMetadata
    hypotheses: list[HypothesisRecord]

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
