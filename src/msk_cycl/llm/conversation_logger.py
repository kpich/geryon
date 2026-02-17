"""Compact session tracing — one JSONL event per significant action."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re

from msk_cycl.lang.results import ComparisonResult
from msk_cycl.llm.generator import HypothesisProposal


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

    def log_tool_call(self, tool_name: str, args: dict, result: str) -> None:
        extra = self._extract_tool_metadata(tool_name, args, result)
        self._write(event="tool_call", tool=tool_name, args=args, **extra)

    def log_proposal(self, idx: int, proposal: HypothesisProposal) -> None:
        filters_a = [f.model_dump() for f in proposal.cycl_spec.query.cohort_a.filters]
        filters_b = [f.model_dump() for f in proposal.cycl_spec.query.cohort_b.filters]
        self._write(
            event="proposal",
            idx=idx,
            cohort_a=proposal.cohort_a_description,
            cohort_b=proposal.cohort_b_description,
            filters_a=filters_a,
            filters_b=filters_b,
        )

    def log_execution(self, idx: int, result: ComparisonResult, time_s: float) -> None:
        self._write(
            event="execution",
            idx=idx,
            success=result.success,
            error=result.error_message,
            cohort_a_size=result.cohort_a_size,
            cohort_b_size=result.cohort_b_size,
            time_s=round(time_s, 3),
        )

    def log_narration(
        self,
        idx: int,
        summary: str,
        tokens: int | None,
        narrative: dict | None = None,
        raw_response: str | None = None,
    ) -> None:
        self._write(event="narration", idx=idx, summary=summary, tokens=tokens)
        if narrative or raw_response:
            self._write_detail(
                event="narration",
                idx=idx,
                narrative=narrative,
                raw_response=raw_response,
            )

    def log_session_end(self, total: int, successful: int, failed: int) -> None:
        self._write(
            event="session_end",
            total_proposals=total,
            successful=successful,
            failed=failed,
        )

    def log_raw_messages(self, messages: list) -> None:
        """Dump full LangGraph conversation to the detail file."""
        for msg in messages:
            msg_type = getattr(msg, "type", type(msg).__name__)
            role = getattr(msg, "role", msg_type)
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None) or []
            record: dict = {
                "event": "message",
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

    @staticmethod
    def _extract_tool_metadata(tool_name: str, args: dict, result: str) -> dict:
        """Pull compact numbers out of the raw tool result string."""
        meta: dict = {}
        if tool_name == "list_tables_tool":
            meta["tables"] = len(result.strip().splitlines())
        elif tool_name == "describe_table_tool":
            m = re.search(r"(\d+)\s+rows", result)
            if m:
                meta["rows"] = int(m.group(1))
            m = re.search(r"(\d+)\s+columns", result)
            if m:
                meta["columns"] = int(m.group(1))
        elif tool_name == "query_data_tool":
            meta["sql"] = args.get("sql", "")
            meta["result_rows"] = len(result.strip().splitlines()) - 1  # minus header
        return meta
