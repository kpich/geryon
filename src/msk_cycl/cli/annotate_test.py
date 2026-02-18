"""Tests for annotation server."""

from io import BytesIO
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from msk_cycl.cli.annotate import (
    _count_all,
    _load_unlabeled,
    _make_handler,
)
from msk_cycl.labeling.labeled_store import LabeledStore
from msk_cycl.labeling.labels import HypothesisRating
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.storage import HypothesisStore
from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import (
    CohortFilter,
    CompareCohorts,
    CyclHyp,
    SelectCohort,
)
from msk_cycl.llm.generator import HypothesisProposal
from msk_cycl.llm.narrator import HypothesisNarrative


def _make_hypothesis(
    hypothesis_id: str = "h-1", session_id: str = "sess-1"
) -> LabeledHypothesis:
    spec = CyclHyp(
        query=CompareCohorts(
            cohort_a=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="TX",
                        operator="==",
                        value="A",
                    )
                ]
            ),
            cohort_b=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient",
                        column="TX",
                        operator="==",
                        value="B",
                    )
                ]
            ),
            outcome=OverallSurvival(),
            method=ComparisonMethod.HAZARD_RATIO_COX,
        )
    )
    return LabeledHypothesis(
        hypothesis_id=hypothesis_id,
        session_id=session_id,
        proposal=HypothesisProposal(
            cohort_a_description="Group A",
            cohort_b_description="Group B",
            outcome_description="Overall survival",
            rationale="Test",
            cycl_spec=spec,
        ),
        spec=spec,
        result=ComparisonResult(
            cohort_a_size=50,
            cohort_b_size=40,
            hazard_ratio=0.72,
            p_value=0.03,
        ),
        execution_time_seconds=1.0,
        narrative=HypothesisNarrative(
            summary="Test summary",
            findings="Test findings",
            limitations=["Limitation"],
            clinical_relevance="Relevant",
        ),
        llm_model="test-model",
    )


def _seed_session(
    output_dir: Path,
    session_id: str,
    hypotheses: list[LabeledHypothesis],
):
    store = HypothesisStore(output_dir / session_id)
    for hyp in hypotheses:
        store.save(hyp)


# -- pure-function tests (fast, real file I/O via tmp_path) --


@pytest.fixture
def setup(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    labeled_dir = tmp_path / "labeled"
    labeled_store = LabeledStore(labeled_dir)

    h1 = _make_hypothesis("h-1", "sess-1")
    h2 = _make_hypothesis("h-2", "sess-1")
    _seed_session(output_dir, "sess-1", [h1, h2])

    return output_dir, labeled_store, labeled_dir


def test_load_unlabeled_returns_all_pending(setup):
    output_dir, labeled_store, _ = setup
    items = _load_unlabeled(output_dir, labeled_store)
    assert len(items) == 2
    ids = {h["hypothesis_id"] for h in items}
    assert ids == {"h-1", "h-2"}


def test_load_unlabeled_excludes_labeled(setup):
    output_dir, labeled_store, _ = setup
    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.rating = HypothesisRating(novelty=2, trustworthiness=3)
    labeled_store.save(hyp)

    items = _load_unlabeled(output_dir, labeled_store)
    assert len(items) == 1
    assert items[0]["hypothesis_id"] == "h-2"


def test_load_unlabeled_includes_critic_rated(setup):
    """Critic-rated hypotheses still appear for human review."""
    output_dir, labeled_store, _ = setup
    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.rating = HypothesisRating(novelty=2, trustworthiness=2)
    hyp.labeled_by = "llm_critic"
    # Re-seed with critic-rated hypothesis
    store = HypothesisStore(output_dir / "sess-1")
    store.save_all([hyp, _make_hypothesis("h-2", "sess-1")])

    items = _load_unlabeled(output_dir, labeled_store)
    ids = {h["hypothesis_id"] for h in items}
    assert "h-1" in ids


def test_load_unlabeled_has_critic_prefill_fields(setup):
    """Critic-rated hypothesis data dict has critic_rating and critic_notes."""
    output_dir, labeled_store, _ = setup
    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.rating = HypothesisRating(novelty=3, uncontrolled=1, trustworthiness=2)
    hyp.labeled_by = "llm_critic"
    hyp.notes = "looks confounded"
    store = HypothesisStore(output_dir / "sess-1")
    store.save_all([hyp])

    items = _load_unlabeled(output_dir, labeled_store)
    assert len(items) == 1
    item = items[0]
    assert item["critic_rating"] == {
        "novelty": 3,
        "uncontrolled": 1,
        "trustworthiness": 2,
        "is_duplicate": None,
        "is_na": None,
    }
    assert item["critic_notes"] == "looks confounded"
    assert item["labeled_by"] == "llm_critic"


def test_count_all(setup):
    output_dir, labeled_store, _ = setup
    counts = _count_all(output_dir, labeled_store)
    assert counts == {"total": 2, "labeled": 0, "pending": 2}

    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.rating = HypothesisRating(novelty=1)
    labeled_store.save(hyp)

    counts = _count_all(output_dir, labeled_store)
    assert counts == {"total": 2, "labeled": 1, "pending": 1}


# -- handler tests (no real server, mocked data layer) --


def _fake_unlabeled(*_args):
    return [
        {"hypothesis_id": "h-1", "summary": "test"},
        {"hypothesis_id": "h-2", "summary": "test2"},
    ]


def _fake_stats(*_args):
    return {"total": 5, "labeled": 2, "pending": 3}


def _invoke_handler(handler_cls, method, path, body=None):
    """Call a handler method without a real socket/server."""
    out = BytesIO()

    handler = handler_cls.__new__(handler_cls)
    handler.path = path
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.client_address = ("127.0.0.1", 0)
    handler.server = MagicMock()
    handler.wfile = out
    handler.close_connection = True
    handler._headers_buffer = []

    if body is not None:
        raw = json.dumps(body).encode()
        handler.headers = {"Content-Length": str(len(raw))}
        handler.rfile = BytesIO(raw)
    else:
        handler.headers = {}
        handler.rfile = BytesIO()

    if method == "GET":
        handler.do_GET()
    elif method == "POST":
        handler.do_POST()

    out.seek(0)
    raw_response = out.read()

    # Parse status from first line, body after \r\n\r\n
    text = raw_response.decode("utf-8", errors="replace")
    status_line = text.split("\r\n")[0]
    status_code = int(status_line.split(" ")[1])
    body_text = text.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in text else ""
    return status_code, body_text


@patch("msk_cycl.cli.annotate._load_unlabeled", _fake_unlabeled)
@patch("msk_cycl.cli.annotate._count_all", _fake_stats)
def test_get_html():
    handler_cls = _make_handler(Path("."), MagicMock())
    status, body = _invoke_handler(handler_cls, "GET", "/")
    assert status == 200
    assert "CYCL Hypothesis Annotator" in body


@patch("msk_cycl.cli.annotate._load_unlabeled", _fake_unlabeled)
@patch("msk_cycl.cli.annotate._count_all", _fake_stats)
def test_api_hypotheses():
    handler_cls = _make_handler(Path("."), MagicMock())
    status, body = _invoke_handler(handler_cls, "GET", "/api/hypotheses")
    assert status == 200
    data = json.loads(body)
    assert len(data) == 2


@patch("msk_cycl.cli.annotate._load_unlabeled", _fake_unlabeled)
@patch("msk_cycl.cli.annotate._count_all", _fake_stats)
def test_api_stats():
    handler_cls = _make_handler(Path("."), MagicMock())
    status, body = _invoke_handler(handler_cls, "GET", "/api/stats")
    assert status == 200
    data = json.loads(body)
    assert data == {"total": 5, "labeled": 2, "pending": 3}


@patch("msk_cycl.cli.annotate._load_unlabeled", _fake_unlabeled)
@patch("msk_cycl.cli.annotate._count_all", _fake_stats)
def test_api_label_saves_hypothesis():
    mock_store = MagicMock()
    hyp = _make_hypothesis("h-1")

    handler_cls = _make_handler(Path("."), mock_store)
    orig_find = handler_cls._find_hypothesis
    handler_cls._find_hypothesis = lambda self, hid: hyp

    try:
        body = {
            "hypothesis_id": "h-1",
            "novelty": 2,
            "trustworthiness": 3,
            "notes": "good",
        }
        status, _ = _invoke_handler(handler_cls, "POST", "/api/label", body=body)
        assert status == 200

        mock_store.save.assert_called_once()
        saved = mock_store.save.call_args[0][0]
        assert saved.rating.novelty == 2
        assert saved.rating.trustworthiness == 3
        assert saved.notes == "good"
        assert saved.labeled_by == "annotator"
    finally:
        handler_cls._find_hypothesis = orig_find


@patch("msk_cycl.cli.annotate._load_unlabeled", _fake_unlabeled)
@patch("msk_cycl.cli.annotate._count_all", _fake_stats)
def test_api_label_saves_na_hypothesis():
    mock_store = MagicMock()
    hyp = _make_hypothesis("h-1")

    handler_cls = _make_handler(Path("."), mock_store)
    orig_find = handler_cls._find_hypothesis
    handler_cls._find_hypothesis = lambda self, hid: hyp

    try:
        body = {
            "hypothesis_id": "h-1",
            "is_na": True,
        }
        status, _ = _invoke_handler(handler_cls, "POST", "/api/label", body=body)
        assert status == 200

        mock_store.save.assert_called_once()
        saved = mock_store.save.call_args[0][0]
        assert saved.rating.is_na is True
        assert saved.rating.novelty is None
        assert saved.rating.uncontrolled is None
        assert saved.rating.trustworthiness is None
        assert saved.labeled_by == "annotator"
    finally:
        handler_cls._find_hypothesis = orig_find
