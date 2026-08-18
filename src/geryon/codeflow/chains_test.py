"""Tests for chain definitions."""

from pathlib import Path

import pytest

from geryon.codeflow.chains import (
    DEFAULT_CHAIN,
    chain_path,
    load_chain,
    validate_chain_name,
)


def _write(chains_dir: Path, name: str, text: str) -> None:
    chains_dir.mkdir(parents=True, exist_ok=True)
    (chains_dir / f"{name}.md").write_text(text)


def test_missing_file_yields_bare_label(tmp_path: Path) -> None:
    chain = load_chain("nonexistent", tmp_path)
    assert chain.name == "nonexistent"
    assert chain.focus is None
    assert chain.data_version is None


def test_frontmatter_and_body(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "medonc-pfs",
        "---\ndata_version: medonc-pfs-2026-08\n---\n\nContrast both PFS defs.\n",
    )
    chain = load_chain("medonc-pfs", tmp_path)
    assert chain.data_version == "medonc-pfs-2026-08"
    assert chain.focus == "Contrast both PFS defs."


def test_body_only_is_focus_with_no_version(tmp_path: Path) -> None:
    _write(tmp_path, "loose", "Just prose, no frontmatter.\n")
    chain = load_chain("loose", tmp_path)
    assert chain.focus == "Just prose, no frontmatter."
    assert chain.data_version is None


def test_frontmatter_only_has_no_focus(tmp_path: Path) -> None:
    _write(tmp_path, "pinned", "---\ndata_version: v1\n---\n")
    chain = load_chain("pinned", tmp_path)
    assert chain.data_version == "v1"
    assert chain.focus is None


def test_unterminated_frontmatter_is_treated_as_prose(tmp_path: Path) -> None:
    _write(tmp_path, "broken", "---\ndata_version: v1\nstill going\n")
    chain = load_chain("broken", tmp_path)
    assert chain.data_version is None
    assert chain.focus is not None and "still going" in chain.focus


def test_unknown_frontmatter_keys_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "extra", "---\nauthor: karl\ndata_version: v2\n---\nprose\n")
    chain = load_chain("extra", tmp_path)
    assert chain.data_version == "v2"
    assert chain.focus == "prose"


@pytest.mark.parametrize("name", ["../escape", "a/b", "", ".", ".."])
def test_validate_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_chain_name(name)


def test_chain_path_uses_given_dir(tmp_path: Path) -> None:
    assert chain_path("medonc-pfs", tmp_path) == tmp_path / "medonc-pfs.md"


def test_default_chain_is_main() -> None:
    assert DEFAULT_CHAIN == "main"
