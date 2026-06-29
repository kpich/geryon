"""Tests for code-hypothesis context formatting."""

from pathlib import Path

from geryon.codeflow.context import (
    format_previous_hypotheses,
    load_prior_hypotheses,
)
from geryon.codeflow.models import CodeHypothesis, CodeNarrative
from geryon.codeflow.store import CodeHypothesisStore
from geryon.sandbox.result import IterationResult


def _hyp(hid: str, session: str = "s1", **kw) -> CodeHypothesis:
    defaults = {
        "hypothesis_id": hid,
        "session_id": session,
        "title": "title",
        "description": "d",
        "rationale": "r",
        "code": "print(1)",
        "success": True,
    }
    defaults.update(kw)
    return CodeHypothesis(**defaults)  # type: ignore[arg-type]


def test_empty_context():
    ctx = format_previous_hypotheses([])
    assert "No previous hypotheses" in ctx.text
    assert ctx.ids == []


def test_context_prefers_context_summary_and_shows_stats():
    h = _hyp(
        "abcdef12",
        result=IterationResult(effect_size=1.83, p_value=0.003),
        narrative=CodeNarrative(
            summary="s", findings="f", context_summary="TP53 worse OS"
        ),
    )
    ctx = format_previous_hypotheses([h])
    assert "TP53 worse OS" in ctx.text
    assert "abcdef12" in ctx.text
    assert "effect=1.83" in ctx.text
    assert "p=0.003" in ctx.text
    assert ctx.ids == ["abcdef12"]


def test_context_marks_failures_and_lineage():
    h = _hyp("child001", success=False, refines="parent99")
    ctx = format_previous_hypotheses([h])
    assert "FAILED" in ctx.text
    assert "refines parent99" in ctx.text


def test_context_newest_first():
    ctx = format_previous_hypotheses([_hyp("old00000"), _hyp("new00000")])
    # newest (last appended) should be listed first
    assert ctx.text.index("new00000") < ctx.text.index("old00000")


def test_load_prior_skips_non_codeflow_and_current_session(tmp_path: Path):
    # A codeflow session from a different session id.
    prior_dir = tmp_path / "2026-06-01" / "sess-prior"
    CodeHypothesisStore(prior_dir).save(_hyp("p1", session="prior"))

    # A non-codeflow jsonl file (legacy-ish) — must be skipped, not crash.
    legacy_dir = tmp_path / "2026-06-02" / "sess-legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "hypotheses.jsonl").write_text(
        '{"record_type": "metadata", "session_id": "legacy"}\n'
        '{"record_type": "hypothesis", "data": {"spec": {"version": 1}}}\n'
    )

    # Current session's own file — must be excluded.
    cur_dir = tmp_path / "2026-06-03" / "sess-cur"
    CodeHypothesisStore(cur_dir).save(_hyp("c1", session="cur"))

    prior = load_prior_hypotheses(tmp_path, current_session_id="cur")
    ids = {h.hypothesis_id for h in prior}
    assert ids == {"p1"}
