"""Tests for the data version marker."""

from pathlib import Path

import pytest

from geryon.etl.data_version import (
    VERSION_MARKER_FILENAME,
    read_data_version,
    resolve_data_version,
    validate_version_name,
    write_version_marker,
)


def test_marker_round_trips(tmp_path: Path) -> None:
    write_version_marker(
        tmp_path, name="medonc-pfs-2026-08", data_root="/src/tree", holdout_seed=42
    )
    assert (tmp_path / VERSION_MARKER_FILENAME).exists()
    assert read_data_version(tmp_path) == "medonc-pfs-2026-08"


def test_read_returns_none_when_unmarked(tmp_path: Path) -> None:
    assert read_data_version(tmp_path) is None


def test_read_returns_none_on_garbage(tmp_path: Path) -> None:
    (tmp_path / VERSION_MARKER_FILENAME).write_text("not json")
    assert read_data_version(tmp_path) is None


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".", ".."])
def test_validate_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_version_name(name)


def test_validate_accepts_reasonable_names() -> None:
    for name in ["2026-06-30", "medonc-pfs-2026-08", "v1.2_beta"]:
        assert validate_version_name(name) == name


def test_resolve_reads_marker_from_split_subdir(tmp_path: Path) -> None:
    """A split dir carries its own copy, so it is self-describing."""
    explore = tmp_path / "explore"
    explore.mkdir()
    write_version_marker(
        explore, name="medonc-pfs-2026-08", data_root="/src", holdout_seed=42
    )
    assert resolve_data_version(explore) == "medonc-pfs-2026-08"


def test_resolve_falls_back_to_parent_marker(tmp_path: Path) -> None:
    write_version_marker(tmp_path, name="named-version", data_root="/s", holdout_seed=1)
    explore = tmp_path / "explore"
    explore.mkdir()
    assert resolve_data_version(explore) == "named-version"


def test_resolve_falls_back_to_dir_name_for_legacy_output(tmp_path: Path) -> None:
    """Legacy dated dirs have no marker; the date is still worth recording."""
    dated = tmp_path / "2026-06-30"
    explore = dated / "explore"
    explore.mkdir(parents=True)
    assert resolve_data_version(explore) == "2026-06-30"
    assert resolve_data_version(dated) == "2026-06-30"
