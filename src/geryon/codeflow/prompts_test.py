"""Tests for focus injection into the system prompts."""

from geryon.codeflow.prompts import (
    CRITIC_FOCUS_NOTE,
    GENERATOR_SYSTEM_PROMPT,
    with_focus,
)


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
