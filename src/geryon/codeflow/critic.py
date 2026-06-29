"""Agentic critic: reviews a code hypothesis and can run code to falsify it.

Unlike the legacy text-only rater, this critic gets the same exploration + sandbox
``run_python`` tools as the generator, so it can actually test a suspicion (e.g.
re-run an analysis adjusting for a confounder) before scoring the hypothesis.
"""

from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent

from geryon.codeflow._shared import (
    build_chat_model,
    make_explore_tools,
    make_run_python_tool,
    sum_message_usage,
)
from geryon.codeflow.models import CodeCritique, CodeHypothesis
from geryon.codeflow.prompts import CRITIC_SYSTEM_PROMPT
from geryon.db import Database
from geryon.llm.caching import (
    cached_text_content,
    supports_cache_control,
    tail_cache_pre_model_hook,
)
from geryon.sandbox import SandboxLimits
from geryon.workflow.session import SessionConfig

if TYPE_CHECKING:
    from geryon.llm.conversation_logger import SessionTracer

_MAX_REACT_CYCLES = 40


class HypothesisCritic:
    """Reviews one hypothesis at a time, optionally running code to test it."""

    def __init__(
        self,
        config: SessionConfig,
        db: Database,
        tracer: SessionTracer | None = None,
        iteration: int = 0,
    ):
        self.config = config
        self.db = db
        self.tracer = tracer
        self.iteration = iteration
        self.limits = SandboxLimits()
        self.llm = build_chat_model(config)
        self.explore_tools = make_explore_tools(db)

    def critique(self, hyp: CodeHypothesis) -> CodeCritique:
        """Return a structured critique, running controls in the sandbox as needed."""
        holder: list[CodeCritique] = []
        tools = self.explore_tools + [
            make_run_python_tool(self.config, self.limits),
            self._make_submit_critique_tool(holder),
        ]
        caching = supports_cache_control(self.config.provider_type)
        graph = create_react_agent(
            self.llm,
            ToolNode(tools, handle_tool_errors=True),
            pre_model_hook=tail_cache_pre_model_hook if caching else None,
        )

        result_block = (
            json.dumps(hyp.result.model_dump(), default=str)
            if hyp.result
            else "(no reported result)"
        )
        user_text = (
            f"# HYPOTHESIS [{hyp.short_id()}]\n{hyp.title}\n{hyp.description}\n\n"
            f"**Rationale**: {hyp.rationale}\n\n"
            f"# RESULT\n{result_block}\n\n"
            f"# CODE\n```python\n{hyp.code}\n```\n\n"
            f"Scrutinize this. Run controls with run_python if you suspect "
            f"confounding, then call submit_critique."
        )

        sys_content: str | list[Any]
        usr_content: str | list[Any]
        if caching:
            sys_content = cached_text_content(CRITIC_SYSTEM_PROMPT)
            usr_content = cached_text_content(user_text)
        else:
            sys_content = CRITIC_SYSTEM_PROMPT
            usr_content = user_text

        steps_per_cycle = 3 if caching else 2
        try:
            result = graph.invoke(
                {
                    "messages": [
                        SystemMessage(content=sys_content),
                        HumanMessage(content=usr_content),
                    ]
                },
                config={"recursion_limit": _MAX_REACT_CYCLES * steps_per_cycle},
            )
            self._log_usage(result.get("messages", []))
        except Exception as e:
            print(f"  ⚠ critic failed for {hyp.short_id()}: {type(e).__name__}: {e}")
            traceback.print_exc()

        if holder:
            return holder[-1]
        # Neutral fallback if the critic never submitted.
        return CodeCritique(
            trustworthiness=2,
            confound_risk=2,
            novelty=2,
            notes="Critic did not submit a structured assessment.",
        )

    def _log_usage(self, messages: list) -> None:
        if self.tracer is None:
            return
        u = sum_message_usage(messages)
        self.tracer.log_generation_usage(
            iteration=self.iteration,
            phase="critic",
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            total_tokens=u.total_tokens,
            cache_read_tokens=u.cache_read_tokens,
            cache_creation_tokens=u.cache_creation_tokens,
            n_llm_calls=u.n_llm_calls,
        )

    def _make_submit_critique_tool(self, holder: list[CodeCritique]):
        @tool
        def submit_critique(
            trustworthiness: int,
            confound_risk: int,
            novelty: int,
            notes: str,
            holds_up: bool | None = None,
            suggested_fix: str | None = None,
            tests_run: list[str] | None = None,
        ) -> str:
            """Record the structured critique. Call exactly once when finished.

            trustworthiness/confound_risk/novelty are each 1-3. Set holds_up only if
            you actually ran a control test. Give suggested_fix when confound_risk>=2.
            """
            critique = CodeCritique(
                trustworthiness=_clamp(trustworthiness),
                confound_risk=_clamp(confound_risk),
                novelty=_clamp(novelty),
                holds_up=holds_up,
                notes=notes,
                suggested_fix=suggested_fix,
                tests_run=tests_run or [],
            )
            holder.append(critique)
            return "✓ critique recorded"

        return submit_critique


def _clamp(value: int) -> int:
    return max(1, min(3, int(value)))
