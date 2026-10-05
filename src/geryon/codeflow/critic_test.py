"""Tests for the critic's structured output and helpers (no LLM/Docker)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import ValidationError
import pytest

from geryon.codeflow.critic import (
    HypothesisCritic,
    _clamp,
    _make_get_search_script_tool,
    format_search,
)
from geryon.codeflow.models import CodeCritique, CodeHypothesis, SearchRun
from geryon.codeflow.store import CodeHypothesisStore
from geryon.sandbox.result import IterationResult
from geryon.workflow.session import SessionConfig


def test_clamp_bounds():
    assert _clamp(0) == 1
    assert _clamp(5) == 3
    assert _clamp(2) == 2


def test_critique_rejects_out_of_range():
    with pytest.raises(ValidationError):
        CodeCritique(trustworthiness=4, confound_risk=2, novelty=1)
    with pytest.raises(ValidationError):
        CodeCritique(trustworthiness=2, confound_risk=0, novelty=1)


def test_critique_defaults():
    c = CodeCritique(trustworthiness=3, confound_risk=1, novelty=2)
    assert c.holds_up is None
    assert c.tests_run == []
    assert c.suggested_fix is None


def test_hypothesis_with_critique_roundtrips(tmp_path):
    store = CodeHypothesisStore(tmp_path)
    hyp = CodeHypothesis(
        hypothesis_id="abc",
        session_id="s",
        title="t",
        description="d",
        rationale="r",
        code="print(1)",
        success=True,
        critique=CodeCritique(
            trustworthiness=2,
            confound_risk=3,
            novelty=2,
            holds_up=False,
            headline="effect vanishes after stage adjustment",
            notes="confounded by stage",
            suggested_fix="adjust for STAGE",
            tests_run=["re-ran adjusting for stage"],
        ),
    )
    store.save(hyp)

    loaded = store.load()[0]
    assert loaded.critique is not None
    assert loaded.critique.confound_risk == 3
    assert loaded.critique.holds_up is False
    assert loaded.critique.headline == "effect vanishes after stage adjustment"
    assert loaded.critique.suggested_fix == "adjust for STAGE"
    assert loaded.critique.tests_run == ["re-ran adjusting for stage"]


def test_critic_that_never_submits_raises_instead_of_inventing_scores(tmp_path):
    config = SessionConfig(
        parquet_dir=Path(tmp_path), storage_dir=Path(tmp_path), enable_llm_logging=False
    )
    graph = MagicMock()
    graph.invoke.return_value = {"messages": []}
    with (
        patch("geryon.codeflow.critic.build_chat_model"),
        patch("geryon.codeflow.critic.make_explore_tools", return_value=[]),
        patch("geryon.codeflow.critic.create_react_agent", return_value=graph),
    ):
        critic = HypothesisCritic(config, db=MagicMock())
        hyp = CodeHypothesis(
            hypothesis_id="abc12345",
            session_id="s",
            title="t",
            description="d",
            rationale="r",
            code="print(1)",
            success=True,
        )
        with pytest.raises(RuntimeError, match="without calling submit_critique"):
            critic.critique(hyp)


def _critique(
    effect: float | None = None, lower: float | None = None, upper: float | None = None
) -> CodeCritique:
    return CodeCritique(
        trustworthiness=2,
        confound_risk=2,
        novelty=2,
        predicted_holdout_effect=effect,
        predicted_holdout_lower=lower,
        predicted_holdout_upper=upper,
    )


def test_forecast_must_be_complete_and_ordered():
    _critique(0.8, 0.6, 1.0)
    with pytest.raises(ValidationError, match="all three"):
        _critique(0.8)
    with pytest.raises(ValidationError, match="<="):
        _critique(1.2, 0.6, 1.0)


def _submit_tool(tmp_path, has_effect):
    config = SessionConfig(
        parquet_dir=Path(tmp_path), storage_dir=Path(tmp_path), enable_llm_logging=False
    )
    with (
        patch("geryon.codeflow.critic.build_chat_model"),
        patch("geryon.codeflow.critic.make_explore_tools", return_value=[]),
    ):
        critic = HypothesisCritic(config, db=MagicMock())
    holder: list[CodeCritique] = []
    return critic._make_submit_critique_tool(holder, has_effect=has_effect), holder


_SCORES = {
    "trustworthiness": 2,
    "confound_risk": 2,
    "novelty": 2,
    "headline": "h",
    "notes": "n",
}
_FORECAST = {
    "predicted_holdout_effect": 0.8,
    "predicted_holdout_lower": 0.6,
    "predicted_holdout_upper": 1.0,
}


def test_submit_requires_forecast_when_effect_reported(tmp_path):
    submit, holder = _submit_tool(tmp_path, has_effect=True)
    assert submit.invoke(_SCORES).startswith("✗")
    assert holder == []
    assert submit.invoke({**_SCORES, **_FORECAST}).startswith("✓")
    assert holder[0].predicted_holdout_effect == 0.8


def test_submit_rejects_forecast_without_effect(tmp_path):
    submit, holder = _submit_tool(tmp_path, has_effect=False)
    assert submit.invoke({**_SCORES, **_FORECAST}).startswith("✗")
    assert submit.invoke(_SCORES).startswith("✓")
    assert holder[0].predicted_holdout_effect is None


def test_submit_returns_bad_interval_to_model(tmp_path):
    submit, holder = _submit_tool(tmp_path, has_effect=True)
    reply = submit.invoke({**_SCORES, **_FORECAST, "predicted_holdout_lower": 0.9})
    assert reply.startswith("✗")
    assert holder == []


# --- the generator's search ----------------------------------------------------

_SEARCH = [
    SearchRun(code="fit(a)", status="OK", output_tail="HR=0.90 p=0.40"),
    SearchRun(code="fit(b)", status="EXIT 1", output_tail="KeyError"),
    SearchRun(
        code="fit(c)",
        status="OK",
        result=IterationResult(effect_size=0.7, p_value=0.04),
        output_tail="HR=0.70 p=0.04",
    ),
]


def test_format_search_lists_every_run():
    text = format_search(_SEARCH)
    assert "3 scripts" in text and "(1 did not exit cleanly)" in text
    assert "## run 1: OK\nHR=0.90 p=0.40" in text
    assert "## run 2: EXIT 1" in text
    assert '"effect_size": 0.7' in text


def test_format_search_says_when_nothing_was_run():
    assert "ran no scripts" in format_search([])


def _critic_user_text(tmp_path, search, sees_search=True) -> tuple[str, list]:
    config = SessionConfig(
        parquet_dir=Path(tmp_path),
        storage_dir=Path(tmp_path),
        enable_llm_logging=False,
        critic_sees_search=sees_search,
    )
    graph = MagicMock()
    graph.invoke.return_value = {"messages": []}
    with (
        patch("geryon.codeflow.critic.build_chat_model"),
        patch("geryon.codeflow.critic.make_explore_tools", return_value=[]),
        patch("geryon.codeflow.critic.ToolNode") as tool_node,
        patch("geryon.codeflow.critic.create_react_agent", return_value=graph),
    ):
        critic = HypothesisCritic(config, db=MagicMock())
        hyp = CodeHypothesis(
            hypothesis_id="abc12345",
            session_id="s",
            title="t",
            description="d",
            rationale="r",
            code="print(1)",
            success=True,
            search=search,
        )
        with pytest.raises(RuntimeError):
            critic.critique(hyp)
    user = graph.invoke.call_args.args[0]["messages"][1]
    text = "".join(block["text"] for block in user.content)
    return text, [t.name for t in tool_node.call_args.args[0]]


def test_critic_is_shown_the_search(tmp_path):
    text, tools = _critic_user_text(tmp_path, _SEARCH)
    assert "# GENERATOR'S SEARCH" in text
    assert "get_search_script" in tools


def test_critic_search_lever_off(tmp_path):
    text, tools = _critic_user_text(tmp_path, _SEARCH, sees_search=False)
    assert "GENERATOR'S SEARCH" not in text
    assert "get_search_script" not in tools


def test_hypothesis_predating_search_shows_no_section(tmp_path):
    text, tools = _critic_user_text(tmp_path, None)
    assert "GENERATOR'S SEARCH" not in text
    assert "get_search_script" not in tools


def test_get_search_script_returns_code_and_rejects_bad_run():
    tool = _make_get_search_script_tool(_SEARCH)
    assert "fit(c)" in tool.invoke({"run": 3})
    assert tool.invoke({"run": 4}).startswith("✗")
