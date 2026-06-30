"""Tests for the code-first runner's exploration-split resolver."""

from pathlib import Path

import pytest

from geryon.codeflow.runner import resolve_explore_dir
from geryon.etl.split_by_patient import SPLIT_MARKER_FILENAME


def _mark(d: Path, split: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / SPLIT_MARKER_FILENAME).write_text(split + "\n")
    return d


def test_resolves_dated_dir_to_explore_subdir(tmp_path: Path):
    _mark(tmp_path / "explore", "explore")
    _mark(tmp_path / "validation", "validation")

    assert resolve_explore_dir(tmp_path) == tmp_path / "explore"


def test_accepts_dir_already_marked_explore(tmp_path: Path):
    explore = _mark(tmp_path / "explore", "explore")

    assert resolve_explore_dir(explore) == explore


def test_rejects_validation_dir(tmp_path: Path):
    validation = _mark(tmp_path / "validation", "validation")

    with pytest.raises(SystemExit, match="validation"):
        resolve_explore_dir(validation)


def test_rejects_legacy_unsplit_dir(tmp_path: Path):
    with pytest.raises(SystemExit, match="Re-run the ETL"):
        resolve_explore_dir(tmp_path)
