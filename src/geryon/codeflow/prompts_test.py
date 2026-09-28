"""Tests for data facts and focus injection into the system prompts."""

from geryon.codeflow.prompts import (
    CRITIC_FOCUS_NOTE,
    GENERATOR_SYSTEM_PROMPT,
    with_data_facts,
    with_focus,
)


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
