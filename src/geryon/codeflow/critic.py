"""Agentic critic: reviews a code hypothesis and can run code to falsify it.

The critic has the same exploration and ``run_python`` tools as the generator, so it
can test a suspicion (e.g. re-run the analysis adjusting for a confounder) before
scoring the hypothesis.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import ValidationError

from geryon.codeflow.agent_tools import make_explore_tools, make_run_python_tool
from geryon.codeflow.chat import build_chat_model, sum_message_usage
from geryon.codeflow.dictionary import data_dictionary_text, make_dictionary_tool
from geryon.codeflow.models import CodeCritique, CodeHypothesis
from geryon.codeflow.prompts import (
    CRITIC_FOCUS_NOTE,
    with_data_dictionary,
    with_focus,
)
from geryon.db import Database
from geryon.llm.caching import (
    cached_text_content,
    tail_cache_pre_model_hook,
)
from geryon.llm.conversation_logger import SessionTracer
from geryon.sandbox import SandboxLimits
from geryon.workflow.session import SessionConfig

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
            make_dictionary_tool(self.config, self.limits, "critic"),
            self._make_submit_critique_tool(
                holder,
                has_effect=hyp.result is not None
                and hyp.result.effect_size is not None,
            ),
        ]
        graph = create_react_agent(
            self.llm,
            ToolNode(tools),
            pre_model_hook=tail_cache_pre_model_hook,
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

        system_prompt = with_focus(
            with_data_dictionary(
                self.config.prompts.critic, data_dictionary_text(self.config)
            ),
            self.config.focus,
            note=CRITIC_FOCUS_NOTE,
        )

        sys_content = cached_text_content(system_prompt)
        usr_content = cached_text_content(user_text)
        result = graph.invoke(
            {
                "messages": [
                    SystemMessage(content=sys_content),
                    HumanMessage(content=usr_content),
                ]
            },
            # Each cycle is three graph steps: pre-model hook, model, tools.
            config={"recursion_limit": _MAX_REACT_CYCLES * 3},
        )
        self._log_trace(result.get("messages", []))

        if not holder:
            # No made-up scores: a critique the critic never gave would read as real.
            raise RuntimeError(
                f"critic finished without calling submit_critique for {hyp.short_id()}"
            )
        return holder[-1]

    def _log_trace(self, messages: list) -> None:
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
        self.tracer.log_raw_messages(messages, phase="critic")

    def _make_submit_critique_tool(self, holder: list[CodeCritique], has_effect: bool):
        @tool
        def submit_critique(
            trustworthiness: int,
            confound_risk: int,
            novelty: int,
            headline: str,
            notes: str,
            holds_up: bool | None = None,
            suggested_fix: str | None = None,
            tests_run: list[str] | None = None,
            predicted_holdout_effect: float | None = None,
            predicted_holdout_lower: float | None = None,
            predicted_holdout_upper: float | None = None,
        ) -> str:
            """Record the structured critique. Call exactly once when finished.

            trustworthiness/confound_risk/novelty are each 1-3. headline is your
            verdict in one short clause (under ~15 words); later iterations see it
            next to this hypothesis. Set holds_up only if you actually ran a control
            test. Give suggested_fix when confound_risk>=2.

            predicted_holdout_effect/_lower/_upper: your forecast of effect_size,
            with an 80% interval, when the unchanged script is rerun on held-out
            patients. Required when the result reports an effect_size; omit
            otherwise.
            """
            forecast = (
                predicted_holdout_effect,
                predicted_holdout_lower,
                predicted_holdout_upper,
            )
            if has_effect and any(f is None for f in forecast):
                return (
                    "✗ The result reports an effect_size, so give "
                    "predicted_holdout_effect, predicted_holdout_lower and "
                    "predicted_holdout_upper. Call submit_critique again."
                )
            if not has_effect and any(f is not None for f in forecast):
                return (
                    "✗ The result reports no effect_size, so there is nothing to "
                    "forecast. Leave the predicted_holdout_* fields out and call "
                    "submit_critique again."
                )
            try:
                critique = CodeCritique(
                    trustworthiness=_clamp(trustworthiness),
                    confound_risk=_clamp(confound_risk),
                    novelty=_clamp(novelty),
                    holds_up=holds_up,
                    headline=headline,
                    notes=notes,
                    suggested_fix=suggested_fix,
                    tests_run=tests_run or [],
                    predicted_holdout_effect=predicted_holdout_effect,
                    predicted_holdout_lower=predicted_holdout_lower,
                    predicted_holdout_upper=predicted_holdout_upper,
                )
            except ValidationError as e:
                return f"✗ {e}. Call submit_critique again."
            holder.append(critique)
            return "✓ critique recorded"

        return submit_critique


def _clamp(value: int) -> int:
    return max(1, min(3, int(value)))
