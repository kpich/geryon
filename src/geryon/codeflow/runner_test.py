"""Tests for the code-first runner's data-dir resolution."""

from pathlib import Path

import pytest

from geryon.codeflow.chains import ChainDef
from geryon.codeflow.runner import (
    get_latest_etl_output,
    resolve_etl_dir,
    resolve_explore_dir,
    resolve_version_dir,
)
from geryon.etl.data_version import write_version_marker
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


# --- data version resolution -------------------------------------------------


def _version(base: Path, name: str, marked: bool = True) -> Path:
    d = base / name
    d.mkdir(parents=True)
    if marked:
        write_version_marker(d, name=name, data_root="/src", holdout_seed=42)
    return d


def test_latest_ignores_named_versions(tmp_path: Path):
    """A named version sorts after every date and must not become 'latest'."""
    _version(tmp_path, "2026-06-18")
    _version(tmp_path, "2026-06-30")
    _version(tmp_path, "medonc-pfs-2026-08")

    assert get_latest_etl_output(tmp_path) == tmp_path / "2026-06-30"


def test_latest_fails_when_only_named_versions_exist(tmp_path: Path):
    _version(tmp_path, "medonc-pfs-2026-08")

    with pytest.raises(FileNotFoundError, match="No dated ETL output dirs"):
        get_latest_etl_output(tmp_path)


def test_resolve_version_dir_reports_a_missing_version(tmp_path: Path):
    with pytest.raises(SystemExit, match="not found"):
        resolve_version_dir(tmp_path, "nope-2026-08")


def test_chain_version_is_used_when_no_flags_given(tmp_path: Path):
    _version(tmp_path, "2026-06-30")
    named = _version(tmp_path, "medonc-pfs-2026-08")
    chain = ChainDef(name="medonc-pfs", data_version="medonc-pfs-2026-08")

    assert resolve_etl_dir(tmp_path, None, None, chain) == named


def test_data_version_flag_overrides_the_chain(tmp_path: Path):
    _version(tmp_path, "medonc-pfs-2026-08")
    other = _version(tmp_path, "medonc-pfs-2026-09")
    chain = ChainDef(name="medonc-pfs", data_version="medonc-pfs-2026-08")

    assert resolve_etl_dir(tmp_path, None, "medonc-pfs-2026-09", chain) == other


def test_falls_back_to_latest_dated_when_nothing_is_named(tmp_path: Path):
    _version(tmp_path, "2026-06-18")
    latest = _version(tmp_path, "2026-06-30")

    assert resolve_etl_dir(tmp_path, None, None, ChainDef(name="main")) == latest


def test_explicit_data_dir_must_still_match_a_named_version(tmp_path: Path):
    """--data-dir cannot quietly point a chain at a different cohort."""
    wrong = _version(tmp_path, "2026-06-30")
    _version(tmp_path, "medonc-pfs-2026-08")
    chain = ChainDef(name="medonc-pfs", data_version="medonc-pfs-2026-08")

    with pytest.raises(SystemExit, match="Data version mismatch"):
        resolve_etl_dir(tmp_path, wrong, None, chain)


def test_explicit_data_dir_is_allowed_when_unmarked(tmp_path: Path):
    """Legacy dirs have no marker; we can't verify, so we don't block."""
    legacy = _version(tmp_path, "2026-01-05", marked=False)
    chain = ChainDef(name="medonc-pfs", data_version="medonc-pfs-2026-08")

    assert resolve_etl_dir(tmp_path, legacy, None, chain) == legacy
