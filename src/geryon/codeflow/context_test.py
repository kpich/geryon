"""Tests for code-hypothesis context formatting."""

import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from geryon.codeflow.context import (
    format_previous_hypotheses,
    load_prior_hypotheses,
)
from geryon.codeflow.models import CodeCritique, CodeHypothesis, CodeNarrative
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
    assert ctx.text == "(none yet)"
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


def test_context_shows_critic_verdict():
    debunked = _hyp(
        "debunk01",
        critique=CodeCritique(
            trustworthiness=1,
            confound_risk=3,
            novelty=2,
            holds_up=False,
            headline="immortal-time bias from ever-IO grouping",
        ),
    )
    held = _hyp(
        "held0001",
        critique=CodeCritique(
            trustworthiness=3, confound_risk=1, novelty=2, holds_up=True
        ),
    )
    ctx = format_previous_hypotheses([debunked, held])
    assert (
        "{critic T1/C3/N2, did NOT hold: immortal-time bias from ever-IO grouping}"
        in ctx.text
    )
    assert "{critic T3/C1/N2, held}" in ctx.text


def test_context_omits_verdict_when_uncritiqued():
    ctx = format_previous_hypotheses([_hyp("nocrit01")])
    line = next(ln for ln in ctx.text.splitlines() if "nocrit01" in ln)
    assert "critic" not in line


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


def test_load_prior_raises_on_unloadable_codeflow_session(tmp_path: Path):
    bad = tmp_path / "2026-06-01" / "sess-bad"
    CodeHypothesisStore(bad).save(_hyp("p1", session="bad"))
    with open(bad / "hypotheses.jsonl", "a") as f:
        f.write('{"record_type": "hypothesis", "data": {"title": 3}}\n')
    with pytest.raises(ValidationError):
        load_prior_hypotheses(tmp_path, current_session_id="cur")


def test_load_prior_raises_on_corrupt_header(tmp_path: Path):
    bad = tmp_path / "2026-06-01" / "sess-bad"
    bad.mkdir(parents=True)
    (bad / "hypotheses.jsonl").write_text("{not json\n")
    with pytest.raises(json.JSONDecodeError):
        load_prior_hypotheses(tmp_path, current_session_id="cur")


def _session(tmp_path: Path, name: str, hid: str, chain: str | None = None) -> None:
    store = (
        CodeHypothesisStore(tmp_path / name)
        if chain is None
        else CodeHypothesisStore(tmp_path / name, chain=chain)
    )
    store.save(_hyp(hid, session=name, chain=chain or "main"))


def test_load_prior_filters_by_chain(tmp_path: Path):
    _session(tmp_path, "s-main", "m1")
    _session(tmp_path, "s-medonc", "d1", chain="medonc-pfs")

    main = load_prior_hypotheses(tmp_path, "cur", chain="main")
    medonc = load_prior_hypotheses(tmp_path, "cur", chain="medonc-pfs")

    assert {h.hypothesis_id for h in main} == {"m1"}
    assert {h.hypothesis_id for h in medonc} == {"d1"}


def test_load_prior_without_chain_loads_every_chain(tmp_path: Path):
    """The unfiltered path backs get_script, which resolves ids across chains."""
    _session(tmp_path, "s-main", "m1")
    _session(tmp_path, "s-medonc", "d1", chain="medonc-pfs")

    every = load_prior_hypotheses(tmp_path, "cur")
    assert {h.hypothesis_id for h in every} == {"m1", "d1"}


def test_pre_chain_sessions_count_as_main(tmp_path: Path):
    """Headers written before chains existed have no chain key."""
    legacy = tmp_path / "2026-06-01" / "sess-old"
    legacy.mkdir(parents=True)
    (legacy / "hypotheses.jsonl").write_text(
        '{"record_type": "metadata", "session_id": "old", "created_at": '
        '"2026-06-01T00:00:00Z", "format": "codeflow"}\n'
        '{"record_type": "hypothesis", "data": '
        + _hyp("old00001", session="old").model_dump_json()
        + "}\n"
    )

    assert {
        h.hypothesis_id for h in load_prior_hypotheses(tmp_path, "cur", "main")
    } == {"old00001"}
    assert load_prior_hypotheses(tmp_path, "cur", chain="medonc-pfs") == []
