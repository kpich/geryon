"""Tests for annotation server."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from geryon.legacy.cli.annotate import (
    _count_all,
    _load_all,
    create_app,
)
from geryon.legacy.labeling.labeled_store import LabeledStore
from geryon.legacy.labeling.labels import HypothesisRating
from geryon.legacy.labeling.models import LabeledHypothesis
from geryon.legacy.labeling.storage import HypothesisStore
from geryon.legacy.lang.methods import ComparisonMethod
from geryon.legacy.lang.outcomes import OverallSurvival
from geryon.legacy.lang.results import ComparisonResult
from geryon.legacy.lang.spec import (
    CohortFilter,
    CompareCohorts,
    GeryonHyp,
    SelectCohort,
)
from geryon.legacy.llm.generator import HypothesisProposal
from geryon.legacy.llm.narrator import HypothesisNarrative


def _make_hypothesis(
    hypothesis_id: str = "h-1", session_id: str = "sess-1"
) -> LabeledHypothesis:
    spec = GeryonHyp(
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
            geryon_spec=spec,
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


def test_load_all_returns_all_hypotheses(setup):
    output_dir, labeled_store, _ = setup
    items = _load_all(output_dir, labeled_store)
    assert len(items) == 2
    ids = {h["hypothesis_id"] for h in items}
    assert ids == {"h-1", "h-2"}
    assert items[0]["created_at"] <= items[1]["created_at"]


def test_load_all_includes_labeled(setup):
    """Labeled hypotheses show with status == 'human_labeled'."""
    output_dir, labeled_store, _ = setup
    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.human_rating = HypothesisRating(novelty=2, trustworthiness=3)
    hyp.labeled_by = "annotator"
    labeled_store.save(hyp)

    items = _load_all(output_dir, labeled_store)
    assert len(items) == 2
    h1 = next(h for h in items if h["hypothesis_id"] == "h-1")
    assert h1["status"] == "human_labeled"


def test_load_all_merges_labeled_store_data(setup):
    """Both critic and human ratings present after merge."""
    output_dir, labeled_store, _ = setup
    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.rating = HypothesisRating(novelty=2, trustworthiness=2)
    hyp.labeled_by = "llm_critic"
    hyp.notes = "critic note"
    store = HypothesisStore(output_dir / "sess-1")
    store.save_all([hyp, _make_hypothesis("h-2", "sess-1")])

    labeled_hyp = _make_hypothesis("h-1", "sess-1")
    labeled_hyp.rating = HypothesisRating(novelty=2, trustworthiness=2)
    labeled_hyp.labeled_by = "llm_critic"
    labeled_hyp.notes = "critic note"
    labeled_hyp.human_rating = HypothesisRating(novelty=3, trustworthiness=3)
    labeled_hyp.human_notes = "human note"
    labeled_store.save(labeled_hyp)

    items = _load_all(output_dir, labeled_store)
    h1 = next(h for h in items if h["hypothesis_id"] == "h-1")
    assert h1["critic_rating"]["novelty"] == 2
    assert h1["human_rating"]["novelty"] == 3
    assert h1["human_notes"] == "human note"
    assert h1["status"] == "human_labeled"


def test_load_all_includes_critic_rated(setup):
    """Critic-rated hypotheses appear with status critic_labeled."""
    output_dir, labeled_store, _ = setup
    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.rating = HypothesisRating(novelty=2, trustworthiness=2)
    hyp.labeled_by = "llm_critic"
    store = HypothesisStore(output_dir / "sess-1")
    store.save_all([hyp, _make_hypothesis("h-2", "sess-1")])

    items = _load_all(output_dir, labeled_store)
    h1 = next(h for h in items if h["hypothesis_id"] == "h-1")
    assert h1["status"] == "critic_labeled"


def test_load_all_has_critic_prefill_fields(setup):
    """Critic-rated hypothesis data dict has critic_rating and critic_notes."""
    output_dir, labeled_store, _ = setup
    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.rating = HypothesisRating(novelty=3, uncontrolled=1, trustworthiness=2)
    hyp.labeled_by = "llm_critic"
    hyp.notes = "looks confounded"
    store = HypothesisStore(output_dir / "sess-1")
    store.save_all([hyp])

    items = _load_all(output_dir, labeled_store)
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


def test_count_all_categories(setup):
    output_dir, labeled_store, _ = setup
    counts = _count_all(output_dir, labeled_store)
    assert counts == {"total": 2, "human_labeled": 0, "critic_labeled": 0, "pending": 2}

    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.human_rating = HypothesisRating(novelty=1)
    hyp.labeled_by = "annotator"
    labeled_store.save(hyp)

    counts = _count_all(output_dir, labeled_store)
    assert counts == {"total": 2, "human_labeled": 1, "critic_labeled": 0, "pending": 1}


# -- Flask route tests (test client, mocked data layer) --


def _fake_all(*_args):
    return [
        {"hypothesis_id": "h-1", "summary": "test", "status": "pending"},
        {"hypothesis_id": "h-2", "summary": "test2", "status": "pending"},
    ]


def _fake_stats(*_args):
    return {"total": 5, "human_labeled": 2, "critic_labeled": 1, "pending": 2}


@pytest.fixture
def client():
    app = create_app(Path("."), MagicMock())
    app.config["TESTING"] = True
    return app.test_client()


@patch("geryon.legacy.cli.annotate._load_all", _fake_all)
@patch("geryon.legacy.cli.annotate._count_all", _fake_stats)
def test_get_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Hypothesis Explorer" in r.data


@patch("geryon.legacy.cli.annotate._load_all", _fake_all)
@patch("geryon.legacy.cli.annotate._count_all", _fake_stats)
def test_api_hypotheses(client):
    r = client.get("/api/hypotheses")
    assert r.status_code == 200
    data = r.get_json()
    assert len(data) == 2


@patch("geryon.legacy.cli.annotate._load_all", _fake_all)
@patch("geryon.legacy.cli.annotate._count_all", _fake_stats)
def test_api_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    data = r.get_json()
    assert data == {"total": 5, "human_labeled": 2, "critic_labeled": 1, "pending": 2}


@patch("geryon.legacy.cli.annotate._load_all", _fake_all)
@patch("geryon.legacy.cli.annotate._count_all", _fake_stats)
def test_api_label_sets_human_rating():
    mock_store = MagicMock()
    mock_store.load_one.return_value = None
    hyp = _make_hypothesis("h-1")
    hyp.rating = HypothesisRating(novelty=1, trustworthiness=1)
    hyp.labeled_by = "llm_critic"
    hyp.notes = "critic note"

    app = create_app(Path("."), mock_store)
    app.config["TESTING"] = True

    with patch("geryon.legacy.cli.annotate._find_hypothesis", return_value=hyp):
        r = app.test_client().post(
            "/api/label",
            json={
                "hypothesis_id": "h-1",
                "novelty": 2,
                "trustworthiness": 3,
                "notes": "good",
            },
        )
        assert r.status_code == 200

    mock_store.save.assert_called_once()
    saved = mock_store.save.call_args[0][0]
    assert saved.human_rating.novelty == 2
    assert saved.human_rating.trustworthiness == 3
    assert saved.human_notes == "good"
    assert saved.labeled_by == "annotator"


@patch("geryon.legacy.cli.annotate._load_all", _fake_all)
@patch("geryon.legacy.cli.annotate._count_all", _fake_stats)
def test_api_label_preserves_critic_rating():
    """Critic values preserved alongside human rating."""
    mock_store = MagicMock()
    mock_store.load_one.return_value = None
    hyp = _make_hypothesis("h-1")
    hyp.rating = HypothesisRating(novelty=1, trustworthiness=1)
    hyp.labeled_by = "llm_critic"
    hyp.notes = "critic note"

    app = create_app(Path("."), mock_store)
    app.config["TESTING"] = True

    with patch("geryon.legacy.cli.annotate._find_hypothesis", return_value=hyp):
        r = app.test_client().post(
            "/api/label",
            json={
                "hypothesis_id": "h-1",
                "novelty": 3,
                "trustworthiness": 3,
                "notes": "human note",
            },
        )
        assert r.status_code == 200

    saved = mock_store.save.call_args[0][0]
    assert saved.rating.novelty == 1
    assert saved.rating.trustworthiness == 1
    assert saved.notes == "critic note"
    assert saved.human_rating.novelty == 3
    assert saved.human_rating.trustworthiness == 3
    assert saved.human_notes == "human note"


@patch("geryon.legacy.cli.annotate._load_all", _fake_all)
@patch("geryon.legacy.cli.annotate._count_all", _fake_stats)
def test_api_label_saves_na_hypothesis():
    mock_store = MagicMock()
    mock_store.load_one.return_value = None
    hyp = _make_hypothesis("h-1")

    app = create_app(Path("."), mock_store)
    app.config["TESTING"] = True

    with patch("geryon.legacy.cli.annotate._find_hypothesis", return_value=hyp):
        r = app.test_client().post(
            "/api/label",
            json={
                "hypothesis_id": "h-1",
                "is_na": True,
            },
        )
        assert r.status_code == 200

    mock_store.save.assert_called_once()
    saved = mock_store.save.call_args[0][0]
    assert saved.human_rating.is_na is True
    assert saved.human_rating.novelty is None
    assert saved.human_rating.uncontrolled is None
    assert saved.human_rating.trustworthiness is None
    assert saved.labeled_by == "annotator"
