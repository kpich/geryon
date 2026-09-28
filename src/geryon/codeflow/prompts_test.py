"""Tests for data facts and focus injection into the system prompts."""

from pathlib import Path
import shutil

import pytest

from geryon.codeflow.prompts import (
    _BUILTIN_SETS_DIR,
    CRITIC_FOCUS_NOTE,
    DEFAULT_PROMPT_SET,
    PREVIOUS_HYPOTHESES_PLACEHOLDER,
    load_prompt_set,
    with_data_facts,
    with_focus,
)

GENERATOR_SYSTEM_PROMPT = load_prompt_set().generator


def test_no_facts_leaves_prompt_untouched() -> None:
    assert with_data_facts(GENERATOR_SYSTEM_PROMPT, None) is GENERATOR_SYSTEM_PROMPT
    assert with_data_facts(GENERATOR_SYSTEM_PROMPT, " \n") is GENERATOR_SYSTEM_PROMPT


def test_facts_are_appended_under_its_heading() -> None:
    out = with_data_facts("BASE", "- [abcd1234] OS runs from first sequencing.")
    assert out.startswith("BASE")
    assert "# Verified facts about this data" in out
    assert "- [abcd1234] OS runs from first sequencing." in out


def test_focus_comes_after_the_facts() -> None:
    out = with_focus(with_data_facts("BASE", "card"), "some focus")
    assert out.index("# Verified facts about this data") < out.index("# Research focus")


def test_no_focus_leaves_prompt_untouched() -> None:
    assert with_focus(GENERATOR_SYSTEM_PROMPT, None) is GENERATOR_SYSTEM_PROMPT
    assert with_focus(GENERATOR_SYSTEM_PROMPT, "") is GENERATOR_SYSTEM_PROMPT
    assert with_focus(GENERATOR_SYSTEM_PROMPT, "   \n ") is GENERATOR_SYSTEM_PROMPT


def test_focus_is_appended_after_the_base_prompt() -> None:
    out = with_focus(GENERATOR_SYSTEM_PROMPT, "Contrast both PFS definitions.")
    assert out.startswith(GENERATOR_SYSTEM_PROMPT)
    assert "Contrast both PFS definitions." in out
    assert "# Research focus" in out


def test_note_is_included_when_given() -> None:
    out = with_focus("BASE", "some focus", note=CRITIC_FOCUS_NOTE)
    assert "some focus" in out
    assert "newly *possible*" in out


def test_note_is_absent_by_default() -> None:
    assert "newly *possible*" not in with_focus("BASE", "some focus")


# --- prompt sets -------------------------------------------------------------


def _copy_default(tmp_path: Path) -> Path:
    dst = tmp_path / "variant"
    shutil.copytree(_BUILTIN_SETS_DIR / DEFAULT_PROMPT_SET, dst)
    return dst


def test_default_set_loads_every_template() -> None:
    ps = load_prompt_set()
    assert ps.name == DEFAULT_PROMPT_SET
    assert ps.generator.startswith("You are a cancer-genomics research agent.")
    assert ps.critic.startswith("You are a skeptical reviewer")
    assert ps.narrator.startswith("You interpret the result")
    assert ps.shows_previous_hypotheses


def test_custom_set_is_loaded_by_path_and_named_by_it(tmp_path: Path) -> None:
    d = _copy_default(tmp_path)
    (d / "generator.md").write_text("VARIANT")
    ps = load_prompt_set(d)
    assert ps.generator == "VARIANT"
    assert ps.name == str(d.resolve())


def test_incomplete_set_is_refused_rather_than_filled_from_default(
    tmp_path: Path,
) -> None:
    d = _copy_default(tmp_path)
    (d / "critic.md").unlink()
    with pytest.raises(FileNotFoundError, match="critic.md"):
        load_prompt_set(d)


def test_missing_set_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt_set(tmp_path / "nope")


def test_user_template_fills_the_placeholder() -> None:
    out = load_prompt_set().render_generator_user("1. [abcd1234] X")
    assert "1. [abcd1234] X" in out
    assert PREVIOUS_HYPOTHESES_PLACEHOLDER not in out


def test_template_without_placeholder_hides_prior_hypotheses(tmp_path: Path) -> None:
    d = _copy_default(tmp_path)
    (d / "generator_user.md").write_text("Generate a hypothesis.")
    assert not load_prompt_set(d).shows_previous_hypotheses
