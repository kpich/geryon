"""Tests for agent helpers that don't require Docker or an LLM."""

from geryon.codeflow.agent import format_run
from geryon.sandbox.result import IterationResult, ScriptRun


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
