"""Code-first hypothesis generation loop (LangGraph ReAct).

The agent explores the data with read-only tools, writes Python that runs in the
Docker sandbox, and submits scripts as hypotheses. Mirrors the legacy
``AutonomousWorkflow`` structure but the deliverable is code, not a formal spec.
"""

from __future__ import annotations

import json
import traceback
from typing import Annotated, Any, NamedTuple, TypedDict
import uuid

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent

from geryon.codeflow._shared import (
    build_chat_model,
    format_run,
    make_explore_tools,
    make_run_python_tool,
    run_in_sandbox,
)
from geryon.codeflow.context import (
    format_previous_hypotheses,
    load_prior_hypotheses,
)
from geryon.codeflow.critic import HypothesisCritic
from geryon.codeflow.models import MAX_STORED_OUTPUT_CHARS, CodeHypothesis
from geryon.codeflow.narrate import CodeNarrator
from geryon.codeflow.prompts import GENERATOR_SYSTEM_PROMPT
from geryon.codeflow.store import CodeHypothesisStore
from geryon.db import Database
from geryon.llm.caching import (
    cached_text_content,
    supports_cache_control,
    tail_cache_pre_model_hook,
)
from geryon.llm.conversation_logger import SessionTracer
from geryon.llm.provider import create_provider
from geryon.llm.schema_context import load_schema_context
from geryon.sandbox import SandboxLimits, ScriptRun, ensure_sandbox
from geryon.workflow.session import Session, SessionConfig

# Re-exported for backwards-compatible imports / tests.
__all__ = ["CodeWorkflow", "format_run"]

# Max model->tools round-trips per iteration (matches legacy budget).
_MAX_REACT_CYCLES = 70


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "Messages in conversation"]


class _GenerationUsage(NamedTuple):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    n_llm_calls: int


def _sum_message_usage(messages: list) -> _GenerationUsage:
    inp = out = tot = cache_read = cache_create = calls = 0
    for msg in messages:
        usage = getattr(msg, "usage_metadata", None)
        if not usage:
            continue
        i = int(usage.get("input_tokens", 0) or 0)
        o = int(usage.get("output_tokens", 0) or 0)
        inp += i
        out += o
        tot += int(usage.get("total_tokens", 0) or 0) or (i + o)
        details = usage.get("input_token_details") or {}
        cache_read += int(details.get("cache_read", 0) or 0)
        cache_create += int(details.get("cache_creation", 0) or 0)
        calls += 1
    return _GenerationUsage(inp, out, tot, cache_read, cache_create, calls)


class CodeWorkflow:
    """LangGraph-based code-first hypothesis generation."""

    def __init__(self, config: SessionConfig):
        self.config = config
        self.session = Session(config)

        self.db = Database(config.parquet_dir)
        provider_kwargs: dict = {"model": config.model}
        if config.provider_type == "aws_bedrock":
            provider_kwargs["region"] = config.aws_region
            provider_kwargs["profile"] = config.aws_profile
        self.provider = create_provider(config.provider_type, **provider_kwargs)

        self.store = CodeHypothesisStore(config.storage_dir)
        self.sandbox_limits = SandboxLimits()

        self.llm_logger: SessionTracer | None = None
        if config.enable_llm_logging:
            self.llm_logger = SessionTracer(
                storage_dir=config.storage_dir,
                session_id=config.session_id,
                model=f"{config.provider_type}/{config.model}",
            )

        self.explore_tools = make_explore_tools(self.db)
        self.llm = build_chat_model(config)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def _run_in_sandbox(self, code: str) -> ScriptRun:
        return run_in_sandbox(self.config, code, self.sandbox_limits)

    def _make_get_script_tool(self, submitted: list[CodeHypothesis]):
        @tool
        def get_script(hypothesis_id: str) -> str:
            """Fetch the full code and result of a previously submitted hypothesis.

            Use this to remix/refine a prior hypothesis. Pass the 8-char id shown in
            the previously-tested list (or a full UUID).
            """
            hyp = self._lookup(hypothesis_id, submitted)
            if hyp is None:
                return f"No hypothesis found matching '{hypothesis_id}'."
            result_block = (
                json.dumps(hyp.result.model_dump(), default=str)
                if hyp.result
                else "(no reported result)"
            )
            return (
                f"# {hyp.title} [{hyp.short_id()}]\n"
                f"{hyp.description}\n\n"
                f"result: {result_block}\n\n"
                f"```python\n{hyp.code}\n```"
            )

        return get_script

    def _make_submit_tool(self, iteration: int, submitted: list[CodeHypothesis]):
        @tool
        def submit(
            title: str,
            description: str,
            rationale: str,
            code: str,
            refines: str | None = None,
        ) -> str:
            """Run a final script and store it as a hypothesis.

            The code is executed in the sandbox; its reported result is recorded
            along with an LLM narrative. Set `refines` to a parent hypothesis id when
            this is a derivation/refinement. Returns a short confirmation.
            """
            print(f"[TOOL] submit called: {title!r}")
            run = self._run_in_sandbox(code)
            parent = self._lookup(refines, submitted) if refines else None

            narrative = None
            try:
                narrator = CodeNarrator(self.provider)
                narrative = narrator.narrate(
                    description=description,
                    rationale=rationale,
                    code=code,
                    result=run.result,
                    stdout=run.stdout,
                )
            except Exception as e:  # narration is best-effort
                print(f"  ⚠ narration failed: {type(e).__name__}: {e}")

            hyp = CodeHypothesis(
                hypothesis_id=str(uuid.uuid4()),
                session_id=self.session.session_id,
                iteration=iteration,
                refines=parent.hypothesis_id if parent else None,
                title=title,
                description=description,
                rationale=rationale,
                code=code,
                result=run.result,
                stdout=run.stdout[:MAX_STORED_OUTPUT_CHARS],
                stderr=run.stderr[:MAX_STORED_OUTPUT_CHARS],
                success=run.success,
                duration_seconds=run.duration_seconds,
                narrative=narrative,
                llm_model=f"{self.config.provider_type}/{self.config.model}",
            )
            self.store.save(hyp)
            submitted.append(hyp)

            headline = (
                narrative.summary
                if narrative and narrative.summary
                else (run.result.summary if run.result and run.result.summary else "")
            )
            status = "OK" if run.success else "SCRIPT FAILED"
            return f"✓ Saved [{hyp.short_id()}] ({status}). {headline}"

        return submit

    def _lookup(
        self, hypothesis_id: str | None, submitted: list[CodeHypothesis]
    ) -> CodeHypothesis | None:
        """Find a hypothesis by full or 8-char id across session + prior sessions."""
        if not hypothesis_id:
            return None
        candidates = submitted + self.store.load() + self._load_prior()
        short = hypothesis_id[:8]
        for hyp in candidates:
            if hyp.hypothesis_id == hypothesis_id or hyp.short_id() == short:
                return hyp
        return None

    def _load_prior(self) -> list[CodeHypothesis]:
        return load_prior_hypotheses(self.config.output_dir, self.config.session_id)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    def run_iteration(
        self, n_proposals: int | None = None, iteration: int | None = None
    ) -> list[CodeHypothesis]:
        if n_proposals is None:
            n_proposals = self.config.num_proposals_per_iteration

        prior = self._load_prior() + self.store.load()
        prev_ctx = format_previous_hypotheses(prior)

        if self.llm_logger and iteration is not None:
            self.llm_logger.log_iteration_start(
                iteration=iteration,
                previous_ids=prev_ctx.ids,
                n_context=len(prev_ctx.ids),
            )

        print(f"Generating {n_proposals} hypothesis(es)...")
        print(f"Using model: {self.config.provider_type}/{self.config.model}")

        schema_ctx = load_schema_context(self.config.parquet_dir, self.db)
        system_content = GENERATOR_SYSTEM_PROMPT
        if schema_ctx:
            system_content = system_content + "\n\n" + schema_ctx
        user_text = (
            f"{prev_ctx.text}\n\n"
            f"Generate {n_proposals} hypothesis(es). Explore first, iterate on your "
            f"script with run_python, then submit."
        )

        caching = supports_cache_control(self.config.provider_type)
        sys_content: str | list[Any]
        usr_content: str | list[Any]
        if caching:
            sys_content = cached_text_content(system_content)
            usr_content = cached_text_content(user_text)
        else:
            sys_content = system_content
            usr_content = user_text
        system_message = SystemMessage(content=sys_content)
        user_message = HumanMessage(content=usr_content)

        submitted: list[CodeHypothesis] = []
        try:
            tools = self.explore_tools + [
                make_run_python_tool(self.config, self.sandbox_limits),
                self._make_get_script_tool(submitted),
                self._make_submit_tool(iteration or 0, submitted),
            ]
            tool_node = ToolNode(tools, handle_tool_errors=True)
            graph = create_react_agent(
                self.llm,
                tool_node,
                pre_model_hook=tail_cache_pre_model_hook if caching else None,
            )

            steps_per_cycle = 3 if caching else 2
            recursion_limit = _MAX_REACT_CYCLES * steps_per_cycle

            result = graph.invoke(
                {"messages": [system_message, user_message]},
                config={"recursion_limit": recursion_limit},
            )

            if self.llm_logger:
                u = _sum_message_usage(result["messages"])
                self.llm_logger.log_generation_usage(
                    iteration=iteration or 0,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    total_tokens=u.total_tokens,
                    cache_read_tokens=u.cache_read_tokens,
                    cache_creation_tokens=u.cache_creation_tokens,
                    n_llm_calls=u.n_llm_calls,
                )
                self.llm_logger.log_raw_messages(result["messages"])

            print(
                f"LLM completed after {len(result['messages'])} messages, "
                f"{len(submitted)} hypotheses submitted"
            )
            if not submitted:
                print("⚠ No hypotheses submitted in this iteration.")
            return submitted

        except Exception as e:
            print("⚠ WARNING: Hypothesis generation failed")
            print(f"  Error: {type(e).__name__}: {e}")
            traceback.print_exc()
            if submitted:
                print(f"  Returning {len(submitted)} submitted before failure.")
            return submitted

    def _critique_and_persist(self, hypotheses: list[CodeHypothesis]) -> None:
        """Run the agentic critic on this iteration's hypotheses and persist results.

        Mutates each hypothesis's ``critique`` in place, then rewrites the store so
        the on-disk records carry the assessments.
        """
        critic = HypothesisCritic(self.config, self.db)
        for hyp in hypotheses:
            print(f"  Critiquing [{hyp.short_id()}]...")
            try:
                hyp.critique = critic.critique(hyp)
            except Exception as e:
                print(f"  ⚠ critic failed [{hyp.short_id()}]: {type(e).__name__}: {e}")

        by_id = {h.hypothesis_id: h for h in hypotheses}
        stored = self.store.load()
        if stored:
            self.store.save_all([by_id.get(s.hypothesis_id, s) for s in stored])

    def run_full_session(self) -> list[CodeHypothesis]:
        ensure_sandbox()  # fail fast if Docker / the image is unavailable

        all_hypotheses: list[CodeHypothesis] = []
        print(f"Starting code-first session {self.session.session_id}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Proposals per iteration: {self.config.num_proposals_per_iteration}")

        for i in range(self.config.max_iterations):
            print(f"=== Iteration {i + 1}/{self.config.max_iterations} ===")
            hypotheses = self.run_iteration(iteration=i + 1)
            if not hypotheses:
                print(f"⚠ No hypotheses in iteration {i + 1}, continuing...")
                continue
            if self.config.critic_cycles > 0:
                self._critique_and_persist(hypotheses)
            all_hypotheses.extend(hypotheses)
            print(f"✓ Iteration {i + 1} complete: {len(hypotheses)} hypotheses")

        if self.llm_logger:
            successful = sum(1 for h in all_hypotheses if h.success)
            self.llm_logger.log_session_end(
                total=len(all_hypotheses),
                successful=successful,
                failed=len(all_hypotheses) - successful,
            )
        print(f"Session complete. Generated {len(all_hypotheses)} hypotheses total.")
        return all_hypotheses
