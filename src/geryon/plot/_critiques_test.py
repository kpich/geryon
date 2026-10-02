"""Tests for refinement depth and critique selection."""

import pytest

from geryon.codeflow.models import CodeCritique, CodeHypothesis
from geryon.plot._critiques import compute_depths, critiqued, score


def _hyp(hid: str, refines: str | None = None, chain: str = "main", crit=True):
    return CodeHypothesis(
        hypothesis_id=hid,
        session_id="s",
        refines=refines,
        chain=chain,
        title="t",
        description="d",
        rationale="r",
        code="",
        success=True,
        critique=CodeCritique(trustworthiness=3, confound_risk=1, novelty=2)
        if crit
        else None,
    )


def test_depths_follow_refines_regardless_of_order():
    hyps = [_hyp("c", refines="b"), _hyp("a"), _hyp("b", refines="a"), _hyp("x")]
    assert compute_depths(hyps) == {"a": 0, "b": 1, "c": 2, "x": 0}


def test_missing_parent_raises():
    with pytest.raises(ValueError, match="not found"):
        compute_depths([_hyp("b", refines="gone")])


def test_score():
    assert score(CodeCritique(trustworthiness=3, confound_risk=1, novelty=2)) == 4


def test_critiqued_indexes_within_chain_and_skips_uncritiqued():
    hyps = [
        _hyp("a"),
        _hyp("o", chain="other"),
        _hyp("b", crit=False),
        _hyp("c"),
    ]
    assert [(i, h.hypothesis_id) for i, h, _ in critiqued(hyps, "main")] == [
        (0, "a"),
        (2, "c"),
    ]
    assert len(critiqued(hyps, None)) == 3
