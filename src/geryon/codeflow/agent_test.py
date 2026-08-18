"""Tests for agent helpers that don't require Docker or an LLM."""

from pathlib import Path
from unittest.mock import patch

from geryon.codeflow.agent import CodeWorkflow, format_run
from geryon.codeflow.models import CodeHypothesis
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
        patch("geryon.codeflow.agent.create_provider"),
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
