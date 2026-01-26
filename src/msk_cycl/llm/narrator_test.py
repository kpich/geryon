"""Tests for result narrator."""

import json
from unittest.mock import Mock

from msk_cycl.llm.narrator import HypothesisNarrative, ResultNarrator


class TestParseNarrative:
    """Tests for _parse_narrative method."""

    def _make_narrator(self) -> ResultNarrator:
        """Create a narrator with mock provider."""
        mock_provider = Mock()
        return ResultNarrator(mock_provider)

    def test_valid_json_parses_correctly(self):
        """Valid JSON parses into HypothesisNarrative."""
        narrator = self._make_narrator()
        content = json.dumps(
            {
                "summary": "Test summary",
                "findings": "Test findings",
                "limitations": ["Limitation 1"],
                "clinical_relevance": "Test relevance",
            }
        )

        result = narrator._parse_narrative(content)

        assert isinstance(result, HypothesisNarrative)
        assert result.summary == "Test summary"
        assert result.findings == "Test findings"
        assert result.limitations == ["Limitation 1"]
        assert result.clinical_relevance == "Test relevance"

    def test_json_in_markdown_code_block_parses(self):
        """JSON wrapped in markdown code blocks parses correctly."""
        narrator = self._make_narrator()
        content = """```json
{
    "summary": "Markdown summary",
    "findings": "Markdown findings",
    "limitations": ["Limit 1", "Limit 2"],
    "clinical_relevance": "Markdown relevance"
}
```"""

        result = narrator._parse_narrative(content)

        assert result.summary == "Markdown summary"
        assert result.findings == "Markdown findings"
        assert result.limitations == ["Limit 1", "Limit 2"]

    def test_invalid_json_returns_fallback(self):
        """Invalid JSON returns fallback narrative."""
        narrator = self._make_narrator()
        content = "This is not valid JSON at all"

        result = narrator._parse_narrative(content)

        assert isinstance(result, HypothesisNarrative)
        assert "Failed to parse LLM response: JSONDecodeError" in result.summary
        assert "This is not valid JSON at all" in result.findings
        assert "Narrative parsing failed" in result.limitations
        assert "Unable to determine due to parsing error" in result.clinical_relevance

    def test_missing_required_fields_returns_fallback(self):
        """Missing required fields returns fallback narrative."""
        narrator = self._make_narrator()
        content = json.dumps(
            {
                "summary": "Only summary",
            }
        )

        result = narrator._parse_narrative(content)

        assert isinstance(result, HypothesisNarrative)
        assert "Failed to parse LLM response: ValidationError" in result.summary
        assert "Narrative parsing failed" in result.limitations

    def test_empty_string_returns_fallback(self):
        """Empty string returns fallback narrative."""
        narrator = self._make_narrator()
        content = ""

        result = narrator._parse_narrative(content)

        assert isinstance(result, HypothesisNarrative)
        assert "Failed to parse LLM response" in result.summary
        assert result.findings == ""

    def test_malformed_backticks_returns_fallback(self):
        """Malformed markdown backticks returns fallback narrative."""
        narrator = self._make_narrator()
        content = "```json\n{incomplete"

        result = narrator._parse_narrative(content)

        assert isinstance(result, HypothesisNarrative)
        assert "Failed to parse LLM response" in result.summary

    def test_long_content_truncated_in_fallback(self):
        """Long content is truncated to 500 chars in fallback findings."""
        narrator = self._make_narrator()
        content = "x" * 1000

        result = narrator._parse_narrative(content)

        assert len(result.findings) == 500

    def test_json_with_extra_fields_parses(self):
        """JSON with extra fields still parses correctly."""
        narrator = self._make_narrator()
        content = json.dumps(
            {
                "summary": "Test summary",
                "findings": "Test findings",
                "limitations": ["Limitation 1"],
                "clinical_relevance": "Test relevance",
                "extra_field": "Should be ignored",
            }
        )

        result = narrator._parse_narrative(content)

        assert result.summary == "Test summary"

    def test_code_block_without_language_parses(self):
        """Code block without language specifier parses correctly."""
        narrator = self._make_narrator()
        content = """```
{
    "summary": "No lang summary",
    "findings": "No lang findings",
    "limitations": [],
    "clinical_relevance": "No lang relevance"
}
```"""

        result = narrator._parse_narrative(content)

        assert result.summary == "No lang summary"
