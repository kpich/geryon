"""LLM conversation logging module."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any

# TODO: Add file locking if we ever parallelize LLM calls
# TODO: Can add logging at OllamaProvider.generate() level for raw HTTP request/response
# if needed


class ConversationLogger:
    """Logger for LLM chat interactions."""

    def __init__(self, storage_dir: Path, session_id: str):
        """Initialize logger.

        Parameters
        ----------
        storage_dir : Path
            Directory to store log files
        session_id : str
            Session identifier (used in filenames)
        """
        self.storage_dir = Path(storage_dir)
        self.session_id = session_id
        self.interaction_count = 0

        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.txt_path = self.storage_dir / f"{session_id}_llm_chat.txt"
        self.json_path = self.storage_dir / f"{session_id}_llm_chat.json"

        # Initialize files
        self._init_txt_log()
        self._init_json_log()

    def _init_txt_log(self) -> None:
        """Initialize human-readable log file."""
        with open(self.txt_path, "w") as f:
            f.write(f"# LLM Chat Log - Session {self.session_id}\n")
            f.write(f"Created: {datetime.utcnow().isoformat()}Z\n\n")

    def _init_json_log(self) -> None:
        """Initialize JSON log file."""
        # JSON file will contain one object per line (JSONL format)
        with open(self.json_path, "w") as f:
            # Write metadata header
            metadata = {
                "type": "metadata",
                "session_id": self.session_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            f.write(json.dumps(metadata) + "\n")

    def log_interaction(
        self,
        interaction_type: str,
        messages: list[dict[str, str]],
        response: dict[str, Any],
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an LLM interaction.

        Parameters
        ----------
        interaction_type : str
            Type of interaction (e.g., "hypothesis_generation", "result_narration")
        messages : list[dict[str, str]]
            Messages sent to LLM (role + content)
        response : dict[str, Any]
            Response from LLM (content + model)
        usage : dict[str, int], optional
            Token usage stats
        metadata : dict[str, Any], optional
            Additional metadata (temperature, etc.)
        """
        self.interaction_count += 1
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Log to human-readable file
        self._append_txt_log(
            interaction_type, timestamp, messages, response, usage, metadata
        )

        # Log to JSON file
        self._append_json_log(
            interaction_type, timestamp, messages, response, usage, metadata
        )

    def _append_txt_log(
        self,
        interaction_type: str,
        timestamp: str,
        messages: list[dict[str, str]],
        response: dict[str, Any],
        usage: dict[str, int] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Append interaction to human-readable log."""
        with open(self.txt_path, "a") as f:
            f.write("---\n\n")
            f.write(f"## Interaction {self.interaction_count}: {interaction_type}\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            # Write messages
            for msg in messages:
                role = msg["role"].title()
                content = msg["content"]
                f.write(f"### {role} Message\n")
                f.write(f"{content}\n\n")

            # Write response
            f.write("### Response\n")
            if "model" in response:
                f.write(f"Model: {response['model']}\n")
            if metadata:
                for key, value in metadata.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
            f.write(f"{response['content']}\n\n")

            # Write usage if available
            if usage:
                f.write(
                    f"**Usage:** {usage.get('prompt_tokens', 0)} prompt tokens, "
                    f"{usage.get('completion_tokens', 0)} completion tokens "
                    f"({usage.get('total_tokens', 0)} total)\n\n"
                )

    def _append_json_log(
        self,
        interaction_type: str,
        timestamp: str,
        messages: list[dict[str, str]],
        response: dict[str, Any],
        usage: dict[str, int] | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Append interaction to JSON log."""
        record = {
            "type": "interaction",
            "interaction_number": self.interaction_count,
            "interaction_type": interaction_type,
            "timestamp": timestamp,
            "messages": messages,
            "response": response,
            "usage": usage,
            "metadata": metadata or {},
        }

        with open(self.json_path, "a") as f:
            f.write(json.dumps(record) + "\n")
