"""Session management for hypothesis generation."""

from datetime import datetime
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    """Configuration for a hypothesis generation session."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # LLM configuration
    provider_type: Literal["ollama", "openai", "anthropic", "aws_bedrock"] = Field(
        default="ollama", description="LLM provider type"
    )
    model: str = Field(default="mixtral:8x7b", description="Model identifier")
    base_url: str | None = Field(
        default=None,
        description="Custom API endpoint (e.g., AWS-hosted inference server)",
    )
    api_key: str | None = Field(
        default=None, description="API key for authentication (optional)"
    )

    # AWS-specific configuration (for aws_bedrock provider)
    aws_region: str | None = Field(
        default="us-east-1", description="AWS region for Bedrock"
    )
    aws_profile: str | None = Field(
        default=None, description="AWS credentials profile name"
    )

    # Generation parameters
    num_proposals_per_iteration: int = Field(
        default=5, description="Hypotheses per iteration"
    )
    max_iterations: int = Field(default=10, description="Maximum iterations to run")

    # Paths
    parquet_dir: Path = Field(..., description="Directory with parquet files")
    storage_dir: Path = Field(..., description="Directory for JSONL output")

    # Logging
    enable_llm_logging: bool = Field(
        default=True, description="Enable LLM conversation logging"
    )


class Session:
    """Manages a hypothesis generation session.

    Tracks session state and provides access to configuration.
    """

    def __init__(self, config: SessionConfig):
        """Initialize session.

        Parameters
        ----------
        config : SessionConfig
            Session configuration
        """
        self.config = config
        self.session_id = config.session_id
        self.created_at = config.created_at
