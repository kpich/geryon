"""Tests for agent helpers that don't require Docker or an LLM."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from geryon.codeflow.agent import CodeWorkflow, format_run
from geryon.codeflow.agent_tools import make_run_python_tool
from geryon.codeflow.chat import MessageUsage
from geryon.codeflow.models import CodeCritique, CodeHypothesis, Expectation, SearchRun
from geryon.codeflow.store import CodeHypothesisStore
from geryon.etl.data_version import write_version_marker
from geryon.etl.split_by_patient import SPLIT_MARKER_FILENAME
from geryon.sandbox.result import IterationResult, ScriptRun
from geryon.workflow.session import SessionConfig


def test_format_run_ok_with_result():
    run = ScriptRun(
        success=True,
        exit_code=0,
        stdout="hello\n",
        duration_seconds=1.5,
        result=IterationResult(effect_size=1.8, p_value=0.01),
    )
    out = format_run(run)
    assert "status: OK" in out
    assert "reported result:" in out
    assert "1.8" in out
    assert "hello" in out


def test_format_run_no_result():
    run = ScriptRun(success=True, exit_code=0, duration_seconds=0.2)
    out = format_run(run)
    assert "did not call report()" in out


def test_format_run_timeout():
    run = ScriptRun(success=False, timed_out=True, error="too slow")
    out = format_run(run)
    assert "status: TIMEOUT" in out
    assert "sandbox error: too slow" in out


def test_format_run_nonzero_exit_shows_stderr():
    run = ScriptRun(success=False, exit_code=1, stderr="Traceback ...")
    out = format_run(run)
    assert "status: EXIT 1" in out
    assert "stderr:" in out
    assert "Traceback" in out


# --- chain isolation ---------------------------------------------------------


def _hyp(hid: str, session: str, chain: str) -> CodeHypothesis:
    return CodeHypothesis(
        hypothesis_id=hid,
        session_id=session,
        chain=chain,
        title="t",
        description="d",
        rationale="r",
        code="print(1)",
        success=True,
    )


def _workflow(tmp_path: Path, chain: str) -> CodeWorkflow:
    """A CodeWorkflow with the LLM/Docker/DuckDB dependencies stubbed out."""
    explore = tmp_path / "data" / "explore"
    explore.mkdir(parents=True)
    (explore / SPLIT_MARKER_FILENAME).write_text("explore\n")
    write_version_marker(
        explore, name="medonc-pfs-2026-08", data_root="/src", holdout_seed=42
    )

    sessions = tmp_path / "sessions"
    config = SessionConfig(
        parquet_dir=explore,
        storage_dir=sessions / "current",
        output_dir=sessions,
        chain=chain,
        enable_llm_logging=False,
    )

    with (
        patch("geryon.codeflow.agent.Database"),
        patch("geryon.codeflow.agent.build_chat_model"),
        patch("geryon.codeflow.agent.make_explore_tools", return_value=[]),
    ):
        return CodeWorkflow(config)


def _seed_other_sessions(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    CodeHypothesisStore(sessions / "s-main", chain="main").save(
        _hyp("mainhyp1", "s-main", "main")
    )
    CodeHypothesisStore(sessions / "s-medonc", chain="medonc-pfs").save(
        _hyp("medonc01", "s-medonc", "medonc-pfs")
    )


def test_prompt_context_sees_only_its_own_chain(tmp_path: Path):
    _seed_other_sessions(tmp_path)
    wf = _workflow(tmp_path, chain="medonc-pfs")

    assert {h.hypothesis_id for h in wf._load_prior()} == {"medonc01"}


def test_get_script_still_resolves_across_chains(tmp_path: Path):
    """The boundary governs what is pushed into the prompt, not what can be pulled."""
    _seed_other_sessions(tmp_path)
    wf = _workflow(tmp_path, chain="medonc-pfs")

    found = wf._lookup("mainhyp1", [])
    assert found is not None
    assert found.chain == "main"


def test_data_version_falls_back_to_the_dirs_marker(tmp_path: Path):
    wf = _workflow(tmp_path, chain="medonc-pfs")
    assert wf.data_version == "medonc-pfs-2026-08"


# --- failures are loud -------------------------------------------------------


class _RaisingGraph:
    def invoke(self, *args, **kwargs):
        raise RuntimeError("ExpiredTokenException")


def test_generation_error_aborts_the_session(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    with (
        patch("geryon.codeflow.agent.ensure_sandbox"),
        patch("geryon.codeflow.agent.create_react_agent", return_value=_RaisingGraph()),
        pytest.raises(RuntimeError, match="ExpiredToken"),
    ):
        wf.run_full_session()


def test_infra_error_inside_a_tool_is_not_fed_to_the_model(tmp_path: Path):
    """A tool that raises (e.g. docker dying mid-submit) must escape the ReAct loop."""
    wf = _workflow(tmp_path, chain="main")
    tool_node = None

    def capture(llm, node, **kwargs):
        nonlocal tool_node
        tool_node = node
        return _RaisingGraph()

    with (
        patch("geryon.codeflow.agent.create_react_agent", side_effect=capture),
        pytest.raises(RuntimeError),
    ):
        wf.run_iteration(iteration=1)

    assert tool_node is not None
    assert tool_node._handle_tool_errors is not True


def test_critic_error_aborts_but_keeps_earlier_critiques(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    first, second = _hyp("aaaa1111", "s", "main"), _hyp("bbbb2222", "s", "main")
    wf.store.save(first)
    wf.store.save(second)
    good = CodeCritique(trustworthiness=3, confound_risk=1, novelty=2)

    with (
        patch("geryon.codeflow.agent.HypothesisCritic") as critic_cls,
        pytest.raises(RuntimeError, match="boom"),
    ):
        critic_cls.return_value.critique.side_effect = [good, RuntimeError("boom")]
        wf._critique_and_persist([first, second])

    stored = {h.hypothesis_id: h for h in wf.store.load()}
    assert stored["aaaa1111"].critique == good
    assert stored["bbbb2222"].critique is None


# --- submit ------------------------------------------------------------------

_SUBMIT_ARGS = {"title": "t", "description": "d", "rationale": "r", "code": "x"}


def _submit(
    wf: CodeWorkflow, run: ScriptRun, search: list[SearchRun] | None = None
) -> tuple[str, list, MagicMock]:
    submitted: list[CodeHypothesis] = []
    with (
        patch.object(wf, "_run_in_sandbox", return_value=run),
        patch("geryon.codeflow.agent.CodeNarrator") as narrator_cls,
    ):
        narrator_cls.return_value.narrate.return_value = None
        narrator_cls.return_value.last_usage = None
        out = wf._make_submit_tool(1, submitted, search or []).invoke(_SUBMIT_ARGS)
    return out, submitted, narrator_cls


def test_submit_rejects_a_crashed_script_and_stores_nothing(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    run = ScriptRun(success=False, exit_code=1, stderr="ValueError: could not convert")

    out, submitted, narrator_cls = _submit(wf, run)

    assert "NOT SAVED" in out
    assert "ValueError: could not convert" in out
    assert submitted == []
    assert wf.store.load() == []
    narrator_cls.assert_not_called()


def test_submit_rejects_an_unparseable_result(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    run = ScriptRun(success=True, exit_code=0, error="result.json is malformed")

    out, submitted, _ = _submit(wf, run)

    assert "NOT SAVED" in out
    assert submitted == []
    assert wf.store.load() == []


def test_submit_stores_a_successful_script(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    run = ScriptRun(success=True, exit_code=0, result=IterationResult(summary="HR 0.6"))

    out, submitted, _ = _submit(wf, run)

    assert out.startswith("✓ Saved")
    assert len(submitted) == 1
    assert [h.hypothesis_id for h in wf.store.load()] == [submitted[0].hypothesis_id]


def test_submit_records_a_blind_expectation_only_with_an_effect(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    expectation = Expectation(question="q", effect=0.8, lower=0.6, upper=1.0)
    usage = MessageUsage(1, 1, 2, 0, 0, 2)
    no_effect = ScriptRun(success=True, exit_code=0, result=IterationResult())
    effect = ScriptRun(
        success=True,
        exit_code=0,
        result=IterationResult(effect_size=0.6, effect_size_type="hazard_ratio"),
    )

    with patch(
        "geryon.codeflow.agent.elicit_expectation", return_value=(expectation, usage)
    ) as elicit:
        _, without, _ = _submit(wf, no_effect)
        elicit.assert_not_called()
        _, with_effect, _ = _submit(wf, effect)

    assert elicit.call_args.kwargs == {
        "title": "t",
        "description": "d",
        "effect_size_type": "hazard_ratio",
    }
    assert without[0].expectation is None
    assert with_effect[0].expectation == expectation
    assert wf.store.load()[1].expectation == expectation


def test_submit_records_the_search_so_far(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    search = [SearchRun(code="print(0.7)", status="OK", output_tail="0.7")]
    run = ScriptRun(success=True, exit_code=0, result=IterationResult(summary="HR 0.6"))

    _, submitted, _ = _submit(wf, run, search)
    search.append(SearchRun(code="later", status="OK"))

    assert [r.code for r in wf.store.load()[0].search or []] == ["print(0.7)"]
    assert len(submitted[0].search or []) == 1


def test_run_python_records_each_run(tmp_path: Path):
    wf = _workflow(tmp_path, chain="main")
    record: list[SearchRun] = []
    tool = make_run_python_tool(wf.config, wf.sandbox_limits, record=record)
    run = ScriptRun(success=False, exit_code=1, stdout="HR=0.71 p=0.04", stderr="boom")

    with patch("geryon.codeflow.agent_tools.run_script", return_value=run):
        tool.invoke({"code": "fit()"})

    assert record == [
        SearchRun(code="fit()", status="EXIT 1", output_tail="HR=0.71 p=0.04\nboom")
    ]
