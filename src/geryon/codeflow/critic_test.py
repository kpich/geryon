"""Tests for the critic's structured output and helpers (no LLM/Docker)."""

from pydantic import ValidationError
import pytest

from geryon.codeflow.critic import _clamp
from geryon.codeflow.models import CodeCritique, CodeHypothesis
from geryon.codeflow.store import CodeHypothesisStore


def test_clamp_bounds():
    assert _clamp(0) == 1
    assert _clamp(5) == 3
    assert _clamp(2) == 2


def test_critique_rejects_out_of_range():
    with pytest.raises(ValidationError):
        CodeCritique(trustworthiness=4, confound_risk=2, novelty=1)
    with pytest.raises(ValidationError):
        CodeCritique(trustworthiness=2, confound_risk=0, novelty=1)


def test_critique_defaults():
    c = CodeCritique(trustworthiness=3, confound_risk=1, novelty=2)
    assert c.holds_up is None
    assert c.tests_run == []
    assert c.suggested_fix is None


def test_hypothesis_with_critique_roundtrips(tmp_path):
    store = CodeHypothesisStore(tmp_path)
    hyp = CodeHypothesis(
        hypothesis_id="abc",
        session_id="s",
        title="t",
        description="d",
        rationale="r",
        code="print(1)",
        success=True,
        critique=CodeCritique(
            trustworthiness=2,
            confound_risk=3,
            novelty=2,
            holds_up=False,
            notes="confounded by stage",
            suggested_fix="adjust for STAGE",
            tests_run=["re-ran adjusting for stage"],
        ),
    )
    store.save(hyp)

    loaded = store.load()[0]
    assert loaded.critique is not None
    assert loaded.critique.confound_risk == 3
    assert loaded.critique.holds_up is False
    assert loaded.critique.suggested_fix == "adjust for STAGE"
    assert loaded.critique.tests_run == ["re-ran adjusting for stage"]
