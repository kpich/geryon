"""JSONL file format schema.

This module defines the structure of JSONL session files for storing labeled
hypotheses. Each file starts with a metadata record, followed by hypothesis
records.

File structure:
    Line 1: SessionFileMetadata record
    Line 2+: HypothesisRecord records

Each line is a complete JSON object. Format evolution via optional fields.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from msk_cycl.labeling.models import LabeledHypothesis


class SessionFileMetadata(BaseModel):
    """Metadata record for JSONL session file (always first line)."""

    record_type: Literal["metadata"] = "metadata"
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(..., description="Session creation timestamp")


class HypothesisRecord(BaseModel):
    """Hypothesis record in JSONL file (one per line after metadata)."""

    record_type: Literal["hypothesis"] = "hypothesis"
    data: LabeledHypothesis = Field(..., description="The labeled hypothesis")
    written_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this record was written",
    )
