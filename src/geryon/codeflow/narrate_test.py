"""Tests for narrator output parsing (no LLM)."""

from unittest.mock import MagicMock

import pytest

from geryon.codeflow.narrate import CodeNarrator


def _narrator() -> CodeNarrator:
    return CodeNarrator(MagicMock())


def test_parse_fenced_json():
    content = '```json\n{"summary": "s", "findings": "f"}\n```'
    assert _narrator()._parse(content).summary == "s"


def test_unparseable_output_raises_instead_of_storing_a_placeholder():
    with pytest.raises(ValueError, match="unparseable"):
        _narrator()._parse("Sure! Here's what I found...")
