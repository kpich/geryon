"""Tests for annotation server API endpoints."""

from http.server import HTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from msk_cycl.cli.annotate import _count_all, _load_unlabeled, _make_handler
from msk_cycl.labeling.labeled_store import LabeledStore
from msk_cycl.labeling.labels import HypothesisLabel
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.storage import HypothesisStore
from msk_cycl.lang.methods import ComparisonMethod
from msk_cycl.lang.outcomes import OverallSurvival
from msk_cycl.lang.results import ComparisonResult
from msk_cycl.lang.spec import CohortFilter, CompareCohorts, CyclHyp, SelectCohort
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
                        table="clinical_patient", column="TX", operator="==", value="A"
                    )
                ]
            ),
            cohort_b=SelectCohort(
                filters=[
                    CohortFilter(
                        table="clinical_patient", column="TX", operator="==", value="B"
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
    output_dir: Path, session_id: str, hypotheses: list[LabeledHypothesis]
):
    store = HypothesisStore(output_dir)
    for hyp in hypotheses:
        store.save(hyp)


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
    hyp.label = HypothesisLabel.CORRECT
    labeled_store.save(hyp)

    items = _load_unlabeled(output_dir, labeled_store)
    assert len(items) == 1
    assert items[0]["hypothesis_id"] == "h-2"


def test_count_all(setup):
    output_dir, labeled_store, _ = setup
    counts = _count_all(output_dir, labeled_store)
    assert counts == {"total": 2, "labeled": 0, "pending": 2}

    hyp = _make_hypothesis("h-1", "sess-1")
    hyp.label = HypothesisLabel.CORRECT
    labeled_store.save(hyp)

    counts = _count_all(output_dir, labeled_store)
    assert counts == {"total": 2, "labeled": 1, "pending": 1}


@pytest.fixture
def server(setup):
    output_dir, labeled_store, labeled_dir = setup
    handler = _make_handler(output_dir, labeled_store)
    srv = HTTPServer(("localhost", 0), handler)
    port = srv.server_address[1]
    thread = Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{port}", labeled_dir
    srv.shutdown()


def test_get_html(server):
    url, _ = server
    resp = urlopen(f"{url}/")
    assert resp.status == 200
    body = resp.read().decode()
    assert "CYCL Hypothesis Annotator" in body


def test_api_hypotheses(server):
    url, _ = server
    resp = urlopen(f"{url}/api/hypotheses")
    data = json.loads(resp.read())
    assert len(data) == 2


def test_api_stats(server):
    url, _ = server
    resp = urlopen(f"{url}/api/stats")
    data = json.loads(resp.read())
    assert data["total"] == 2
    assert data["pending"] == 2


def test_api_label_creates_file(server):
    url, labeled_dir = server
    payload = json.dumps(
        {"hypothesis_id": "h-1", "label": "correct", "notes": "looks good"}
    ).encode()
    req = Request(
        f"{url}/api/label",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urlopen(req)
    assert resp.status == 200

    labeled_file = labeled_dir / "h-1.json"
    assert labeled_file.exists()

    content = json.loads(labeled_file.read_text())
    assert content["label"] == "correct"
    assert content["label_notes"] == "looks good"
    assert content["labeled_by"] == "annotator"


def test_api_label_removes_from_pending(server):
    url, labeled_dir = server
    payload = json.dumps({"hypothesis_id": "h-1", "label": "red_herring"}).encode()
    req = Request(
        f"{url}/api/label",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urlopen(req)

    resp = urlopen(f"{url}/api/hypotheses")
    data = json.loads(resp.read())
    assert len(data) == 1
    assert data[0]["hypothesis_id"] == "h-2"
