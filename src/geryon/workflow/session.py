"""Session management for hypothesis generation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, Field

from geryon.codeflow.chains import DEFAULT_CHAIN
from geryon.codeflow.prompts import PromptSet, load_prompt_set

DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-opus-5-5"
# Older models lack the classifiers that stop Opus 5.5 on some oncology text.
DEFAULT_FALLBACK_MODEL = "us.anthropic.claude-opus-4-8"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class SessionConfig(BaseModel):
    """Configuration for a hypothesis generation session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # LLM configuration
    model: str = Field(
        default=DEFAULT_BEDROCK_MODEL,
        description="Model identifier",
    )
    fallback_model: str = Field(
        default=DEFAULT_FALLBACK_MODEL,
        description="Model a call is re-sent to when the primary model's safety "
        "classifier refuses it (stop_reason 'refusal')",
    )
    # Sent explicitly rather than left to the model's default, which differs by model
    # (medium on Opus 5.5, high before it) and could change under us.
    effort: Effort = Field(
        default="medium",
        description="Claude output_config.effort for generator, critic and narrator: "
        "how much the model thinks",
    )

    # AWS Bedrock
    aws_region: str | None = Field(
        default="us-east-2", description="AWS region for Bedrock"
    )
    aws_profile: str | None = Field(
        default=None, description="AWS credentials profile name"
    )

    # Generation parameters
    max_iterations: int = Field(default=10, description="Maximum iterations to run")
    sandbox_timeout_seconds: int = Field(
        default=180,
        description="Wall-clock seconds before a sandboxed script is killed (codeflow)",
    )
    run_critic: bool = Field(
        default=True, description="Critique each hypothesis with the agentic critic"
    )

    # Line of investigation. Prior hypotheses are injected from this chain only, and the
    # focus prose (from chains/<chain>.md) steers generator, critic and narrator.
    chain: str = Field(default=DEFAULT_CHAIN, description="Chain this session extends")
    focus: str | None = Field(
        default=None,
        description="Resolved focus prose appended to the system prompts",
    )
    data_version: str | None = Field(
        default=None,
        description="Human-readable name of the data version being read",
    )

    # Steering levers, recorded per session so runs with different settings can be
    # compared.
    prompts: PromptSet = Field(
        default_factory=load_prompt_set,
        description="Resolved prompt texts (see geryon.codeflow.prompts)",
    )
    include_data_dictionary: bool = Field(
        default=True, description="Show the data dictionary in all three prompts"
    )
    code_version: str | None = Field(
        default=None,
        description="git commit of the geryon checkout, '+dirty' if it had changes",
    )

    # Paths
    parquet_dir: Path = Field(..., description="Directory with parquet files")
    storage_dir: Path = Field(..., description="Directory for JSONL output")
    output_dir: Path | None = Field(
        default=None,
        description="Parent output directory (for loading prior sessions)",
    )

    # Logging
    enable_llm_logging: bool = Field(
        default=True, description="Enable LLM conversation logging"
    )

    def to_config_dict(self) -> dict:
        return self.model_dump(mode="json")


class Session:
    def __init__(self, config: SessionConfig):
        self.config = config
        self.session_id = config.session_id
        self.created_at = config.created_at
