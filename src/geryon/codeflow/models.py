"""Data model for a code-first hypothesis.

Replaces the spec-bound ``LabeledHypothesis``: the deliverable is the Python
``code`` plus its standardized ``IterationResult`` and run metadata. Lineage is the
``refines`` parent-pointer (deriving = remix the parent's code).
"""

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from geryon.codeflow.chains import DEFAULT_CHAIN
from geryon.sandbox.result import IterationResult

# Cap stored stdout/stderr so a chatty script can't bloat the JSONL. The full
# transcript still lands in the tracer's detail log.
MAX_STORED_OUTPUT_CHARS = 20_000


class CodeNarrative(BaseModel):
    """LLM interpretation of a code hypothesis's result."""

    summary: str
    findings: str
    limitations: list[str] = Field(default_factory=list)
    # 1-2 sentences reinjected into later prompts so context doesn't carry full history.
    context_summary: str | None = None


class CodeCritique(BaseModel):
    """An agentic critique of a code hypothesis.

    The critic can run code to test its suspicions, so ``holds_up`` records whether
    the effect survived a control the critic actually tried (None if not tested).
    """

    trustworthiness: int = Field(..., ge=1, le=3, description="1=weak, 3=solid")
    confound_risk: int = Field(
        ..., ge=1, le=3, description="1=low, 3=highly confounded"
    )
    novelty: int = Field(..., ge=1, le=3, description="1=trivial, 3=novel")
    holds_up: bool | None = Field(
        default=None, description="Did the effect survive a control the critic ran?"
    )
    notes: str = ""
    suggested_fix: str | None = None
    tests_run: list[str] = Field(
        default_factory=list, description="Short descriptions of checks the critic ran"
    )


class CodeHypothesis(BaseModel):
    """A single code-first hypothesis: proposal → execution → narrative."""

    # Identity / lineage
    hypothesis_id: str = Field(..., description="Unique identifier (UUID)")
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    iteration: int | None = Field(default=None, description="1-indexed iteration")
    refines: str | None = Field(
        default=None, description="Parent hypothesis_id this remixes, if any"
    )
    chain: str = Field(
        default=DEFAULT_CHAIN, description="Line of investigation this belongs to"
    )
    data_version: str | None = Field(
        default=None,
        description="Human-readable name of the cohort version this was computed on",
    )

    # Proposal / deliverable
    title: str = Field(..., description="Short title of the hypothesis")
    description: str = Field(..., description="What the analysis tests, in prose")
    rationale: str = Field(
        ..., description="Why it's worth testing / how it's controlled"
    )
    code: str = Field(..., description="The Python script that was executed")

    # Execution
    result: IterationResult | None = Field(
        default=None, description="Standardized result the script reported"
    )
    stdout: str = ""
    stderr: str = ""
    success: bool = Field(..., description="Script exited cleanly")
    duration_seconds: float = 0.0

    # Narration
    narrative: CodeNarrative | None = None
    llm_model: str = ""

    # Critique (agentic critic that can run code to test suspicions)
    critique: CodeCritique | None = None

    # Labeling (human/critic), optional
    notes: str | None = None

    def short_id(self) -> str:
        return self.hypothesis_id[:8]
