"""Compact session tracing — one JSONL event per significant action."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path


class SessionTracer:
    """Writes a compact trace.jsonl event log per run directory."""

    def __init__(self, storage_dir: Path, session_id: str, model: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.storage_dir / "trace.jsonl"
        self.detail_path = self.storage_dir / "detail.jsonl"

        self._write(
            event="session_start",
            session_id=session_id,
            model=model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_iteration_start(
        self,
        iteration: int,
        previous_ids: list[str],
        n_context: int,
    ) -> None:
        self._write(
            event="iteration_start",
            iteration=iteration,
            previous_hypothesis_ids=previous_ids,
            n_context=n_context,
        )

    def log_generation_usage(
        self,
        iteration: int,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        n_llm_calls: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        phase: str = "generation",
    ) -> None:
        """Token usage for one LLM phase of an iteration.

        ``phase`` is ``generation``, ``critic`` or ``narration``. All phases share the
        ``generation_usage`` event so the cost plot sums the whole run.
        ``input_tokens`` is uncached input only; the two cache counts are separate
        from it.
        """
        self._write(
            event="generation_usage",
            iteration=iteration,
            phase=phase,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            n_llm_calls=n_llm_calls,
        )

    def log_session_end(self, total: int, successful: int, failed: int) -> None:
        self._write(
            event="session_end",
            total_proposals=total,
            successful=successful,
            failed=failed,
        )

    def log_raw_messages(self, messages: list, phase: str = "generation") -> None:
        """Dump a full LangGraph conversation to the detail file.

        ``phase`` tags each message so the generator and critic transcripts can
        be told apart in detail.jsonl.
        """
        for msg in messages:
            msg_type = getattr(msg, "type", type(msg).__name__)
            role = getattr(msg, "role", msg_type)
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None) or []
            record: dict = {
                "event": "message",
                "phase": phase,
                "role": role,
                "type": msg_type,
                "content": content,
            }
            if tool_calls:
                record["tool_calls"] = [
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                    }
                    for tc in tool_calls
                ]
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                record["tool_call_id"] = tool_call_id
            name = getattr(msg, "name", None)
            if name:
                record["name"] = name
            self._write_detail(**record)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write(self, **fields: object) -> None:
        if "ts" not in fields:
            fields["ts"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(fields, default=str) + "\n")

    def _write_detail(self, **fields: object) -> None:
        if "ts" not in fields:
            fields["ts"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self.detail_path, "a") as f:
            f.write(json.dumps(fields, default=str) + "\n")
