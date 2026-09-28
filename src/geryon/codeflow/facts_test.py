"""Tests for the verified data-facts store and the record_data_fact tool."""

from pathlib import Path
from unittest.mock import patch

from geryon.codeflow._shared import data_facts_text, make_record_fact_tool
from geryon.codeflow.facts import DataFact, DataFactStore, format_facts, has_assert
from geryon.sandbox import SandboxLimits, ScriptRun
from geryon.workflow.session import SessionConfig

CHECK = "from geryon_runtime import db\nassert 1 == 1\n"


def _fact(text: str, version: str = "v1", supersedes: str | None = None) -> DataFact:
    return DataFact(
        fact=text,
        check_code=CHECK,
        data_version=version,
        session_id="s",
        recorded_by="generator",
        supersedes=supersedes,
    )


def _config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(
        parquet_dir=tmp_path / "explore",
        storage_dir=tmp_path / "sessions" / "current",
        output_dir=tmp_path / "sessions",
        data_version="v1",
        enable_llm_logging=False,
    )


def test_has_assert() -> None:
    assert has_assert(CHECK)
    assert not has_assert("print(1)")
    assert not has_assert("assert (")


def test_current_filters_by_version_and_drops_superseded(tmp_path: Path) -> None:
    store = DataFactStore(tmp_path)
    old = _fact("OS runs from diagnosis")
    other = _fact("other version", version="v2")
    store.append(old)
    store.append(other)
    store.append(_fact("OS runs from sequencing", supersedes=old.fact_id))

    assert [f.fact for f in store.current("v1")] == ["OS runs from sequencing"]
    assert [f.fact for f in store.current("v2")] == ["other version"]


def test_format_facts_is_none_when_empty() -> None:
    assert format_facts([]) is None
    f = _fact("x")
    assert format_facts([f]) == f"- [{f.short_id()}] x"


def _record(tmp_path: Path, run: ScriptRun, **kwargs) -> str:
    config = _config(tmp_path)
    tool = make_record_fact_tool(config, SandboxLimits(), "critic")
    with patch("geryon.codeflow._shared.run_in_sandbox", return_value=run) as sandbox:
        out = tool.invoke({"fact": "f", "check_code": CHECK, **kwargs})
    kwargs.setdefault("sandbox", sandbox)
    return out


def test_passing_check_saves_the_fact(tmp_path: Path) -> None:
    out = _record(tmp_path, ScriptRun(success=True, exit_code=0))
    assert out.startswith("✓ Fact saved")
    saved = DataFactStore(tmp_path / "sessions").load_all()
    assert [(f.fact, f.recorded_by, f.data_version) for f in saved] == [
        ("f", "critic", "v1")
    ]
    assert data_facts_text(_config(tmp_path)) == f"- [{saved[0].short_id()}] f"


def test_failing_check_saves_nothing(tmp_path: Path) -> None:
    out = _record(tmp_path, ScriptRun(success=False, exit_code=1, stderr="Assert"))
    assert out.startswith("✗ NOT SAVED: the check failed")
    assert DataFactStore(tmp_path / "sessions").load_all() == []


def test_check_without_assert_is_rejected_before_running(tmp_path: Path) -> None:
    tool = make_record_fact_tool(_config(tmp_path), SandboxLimits(), "generator")
    with patch("geryon.codeflow._shared.run_in_sandbox") as sandbox:
        out = tool.invoke({"fact": "f", "check_code": "print(1)"})
    assert "no assert" in out
    sandbox.assert_not_called()


def test_unknown_supersedes_is_rejected(tmp_path: Path) -> None:
    out = _record(tmp_path, ScriptRun(success=True, exit_code=0), supersedes="nope")
    assert "no fact with id 'nope'" in out
    assert DataFactStore(tmp_path / "sessions").load_all() == []
