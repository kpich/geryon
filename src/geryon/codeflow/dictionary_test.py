"""Tests for the data dictionary store and the add_to_data_dictionary tool."""

from pathlib import Path
from unittest.mock import patch

from geryon.codeflow._shared import data_dictionary_text, make_dictionary_tool
from geryon.codeflow.dictionary import (
    DataDictionary,
    DictionaryEntry,
    format_entries,
    has_assert,
)
from geryon.sandbox import SandboxLimits, ScriptRun
from geryon.workflow.session import SessionConfig

CHECK = "from geryon_runtime import db\nassert 1 == 1\n"


def _entry(
    text: str, version: str = "v1", supersedes: str | None = None
) -> DictionaryEntry:
    return DictionaryEntry(
        entry=text,
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
    store = DataDictionary(tmp_path)
    old = _entry("OS runs from diagnosis")
    other = _entry("other version", version="v2")
    store.append(old)
    store.append(other)
    store.append(_entry("OS runs from sequencing", supersedes=old.entry_id))

    assert [f.entry for f in store.current("v1")] == ["OS runs from sequencing"]
    assert [f.entry for f in store.current("v2")] == ["other version"]


def test_format_entries_is_none_when_empty() -> None:
    assert format_entries([]) is None
    f = _entry("x")
    assert format_entries([f]) == f"- [{f.short_id()}] x"


def _record(tmp_path: Path, run: ScriptRun, **kwargs) -> str:
    config = _config(tmp_path)
    tool = make_dictionary_tool(config, SandboxLimits(), "critic")
    with patch("geryon.codeflow._shared.run_in_sandbox", return_value=run) as sandbox:
        out = tool.invoke({"entry": "f", "check_code": CHECK, **kwargs})
    kwargs.setdefault("sandbox", sandbox)
    return out


def test_passing_check_saves_the_entry(tmp_path: Path) -> None:
    out = _record(tmp_path, ScriptRun(success=True, exit_code=0))
    assert out.startswith("✓ Entry saved")
    saved = DataDictionary(tmp_path / "sessions").load_all()
    assert [(f.entry, f.recorded_by, f.data_version) for f in saved] == [
        ("f", "critic", "v1")
    ]
    assert data_dictionary_text(_config(tmp_path)) == f"- [{saved[0].short_id()}] f"


def test_failing_check_saves_nothing(tmp_path: Path) -> None:
    out = _record(tmp_path, ScriptRun(success=False, exit_code=1, stderr="Assert"))
    assert out.startswith("✗ NOT SAVED: the check failed")
    assert DataDictionary(tmp_path / "sessions").load_all() == []


def test_check_without_assert_is_rejected_before_running(tmp_path: Path) -> None:
    tool = make_dictionary_tool(_config(tmp_path), SandboxLimits(), "generator")
    with patch("geryon.codeflow._shared.run_in_sandbox") as sandbox:
        out = tool.invoke({"entry": "f", "check_code": "print(1)"})
    assert "no assert" in out
    sandbox.assert_not_called()


def test_unknown_supersedes_is_rejected(tmp_path: Path) -> None:
    out = _record(tmp_path, ScriptRun(success=True, exit_code=0), supersedes="nope")
    assert "no entry with id 'nope'" in out
    assert DataDictionary(tmp_path / "sessions").load_all() == []


def test_dictionary_can_be_left_out_of_the_prompts(tmp_path: Path) -> None:
    _record(tmp_path, ScriptRun(success=True, exit_code=0))
    config = _config(tmp_path).model_copy(update={"include_data_dictionary": False})
    assert data_dictionary_text(config) is None
