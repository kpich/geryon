"""Browser-based annotation server for labeling hypotheses.

Flask app that serves a template and JSON API. Lets reviewers rate
hypotheses on multiple dimensions instead of typing label strings
at a terminal prompt.
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
import webbrowser

from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError

from geryon.legacy.labeling.labeled_store import LabeledStore
from geryon.legacy.labeling.labels import RATING_DIMENSIONS, HypothesisRating
from geryon.legacy.labeling.models import LabeledHypothesis
from geryon.legacy.labeling.storage import HypothesisStore


def _serialize_hypothesis(
    hyp: LabeledHypothesis,
    labeled_data: LabeledHypothesis | None,
) -> dict:
    """Build per-hypothesis dict, merging labeled-store data."""
    hid = hyp.hypothesis_id

    # Determine status
    has_human = (
        labeled_data is not None
        and labeled_data.human_rating is not None
        and not labeled_data.human_rating.is_pending
    )
    if has_human:
        status = "human_labeled"
    elif hyp.labeled_by == "llm_critic" and not hyp.rating.is_pending:
        status = "critic_labeled"
    elif labeled_data is not None and labeled_data.labeled_by == "annotator":
        status = "human_labeled"
    else:
        status = "pending"

    human_rating = None
    human_notes = None
    if labeled_data is not None:
        if labeled_data.human_rating is not None:
            human_rating = {
                "novelty": labeled_data.human_rating.novelty,
                "uncontrolled": labeled_data.human_rating.uncontrolled,
                "trustworthiness": labeled_data.human_rating.trustworthiness,
                "is_duplicate": labeled_data.human_rating.is_duplicate,
                "is_na": labeled_data.human_rating.is_na,
            }
        human_notes = labeled_data.human_notes

    refines = hyp.proposal.refines_hypothesis

    return {
        "hypothesis_id": hid,
        "session_id": hyp.session_id,
        "created_at": str(hyp.created_at),
        "cohort_a_description": hyp.proposal.cohort_a_description,
        "cohort_b_description": hyp.proposal.cohort_b_description,
        "outcome_description": hyp.proposal.outcome_description,
        "rationale": hyp.proposal.rationale,
        "cohort_a_size": hyp.result.cohort_a_size,
        "cohort_b_size": hyp.result.cohort_b_size,
        "hazard_ratio": hyp.result.hazard_ratio,
        "confidence_interval_lower": hyp.result.confidence_interval_lower,
        "confidence_interval_upper": hyp.result.confidence_interval_upper,
        "p_value": hyp.result.p_value,
        "median_a": hyp.result.median_a,
        "median_b": hyp.result.median_b,
        "u_statistic": hyp.result.u_statistic,
        "spec": hyp.spec.model_dump(),
        "summary": hyp.narrative.summary,
        "findings": hyp.narrative.findings,
        "limitations": hyp.narrative.limitations,
        "clinical_relevance": hyp.narrative.clinical_relevance,
        "iteration": hyp.iteration,
        "labeled_by": hyp.labeled_by,
        "status": status,
        "refines_hypothesis": refines,
        "refines_hypothesis_short": refines[:8] if refines else None,
        "critic_rating": (
            {
                "novelty": hyp.rating.novelty,
                "uncontrolled": hyp.rating.uncontrolled,
                "trustworthiness": hyp.rating.trustworthiness,
                "is_duplicate": hyp.rating.is_duplicate,
                "is_na": hyp.rating.is_na,
            }
            if hyp.labeled_by == "llm_critic"
            else None
        ),
        "critic_notes": (hyp.notes if hyp.labeled_by == "llm_critic" else None),
        "human_rating": human_rating,
        "human_notes": human_notes,
    }


def _load_all(output_dir: Path, labeled_store: LabeledStore) -> list[dict]:
    """Load ALL hypotheses, merge labeled-store data."""
    seen: dict[str, dict] = {}
    for jsonl_file in sorted(output_dir.rglob("hypotheses.jsonl")):
        store = HypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load()
        except Exception:
            continue

        for hyp in hypotheses:
            hid = hyp.hypothesis_id
            if hid in seen:
                continue
            labeled_data = labeled_store.load_one(hid)
            seen[hid] = _serialize_hypothesis(hyp, labeled_data)

    items = list(seen.values())
    items.sort(key=lambda x: x["created_at"])
    return items


def _count_all(output_dir: Path, labeled_store: LabeledStore) -> dict:
    """Return total / human_labeled / critic_labeled / pending counts."""
    all_hyps = _load_all(output_dir, labeled_store)
    total = len(all_hyps)
    human_labeled = sum(1 for h in all_hyps if h["status"] == "human_labeled")
    critic_labeled = sum(1 for h in all_hyps if h["status"] == "critic_labeled")
    pending = sum(1 for h in all_hyps if h["status"] == "pending")
    return {
        "total": total,
        "human_labeled": human_labeled,
        "critic_labeled": critic_labeled,
        "pending": pending,
    }


def _find_hypothesis(output_dir: Path, hypothesis_id: str):
    """Find a hypothesis by ID across all session stores."""
    for jsonl_file in output_dir.rglob("hypotheses.jsonl"):
        store = HypothesisStore(jsonl_file.parent)
        try:
            for hyp in store.load():
                if hyp.hypothesis_id == hypothesis_id:
                    return hyp
        except Exception:
            continue
    return None


def create_app(output_dir: Path, labeled_store: LabeledStore) -> Flask:
    """App factory for the annotation server."""
    app = Flask(__name__)

    dimensions = {
        k: {
            "label": v["label"],
            "levels": {str(lk): lv for lk, lv in v["levels"].items()},
        }
        for k, v in RATING_DIMENSIONS.items()
    }

    @app.get("/")
    def index():
        return render_template("annotate.html", dimensions=dimensions)

    @app.get("/api/hypotheses")
    def api_hypotheses():
        return jsonify(_load_all(output_dir, labeled_store))

    @app.get("/api/stats")
    def api_stats():
        return jsonify(_count_all(output_dir, labeled_store))

    @app.post("/api/label")
    def api_label():
        body = request.get_json()

        hid = body.get("hypothesis_id")
        if not hid:
            return "Missing hypothesis_id", 400

        try:
            rating = HypothesisRating(
                novelty=body.get("novelty"),
                uncontrolled=body.get("uncontrolled"),
                trustworthiness=body.get("trustworthiness"),
                is_duplicate=body.get("is_duplicate"),
                is_na=body.get("is_na"),
            )
        except ValidationError as exc:
            return str(exc), 400

        if rating.is_pending:
            return "At least one dimension must be rated", 400

        hyp = _find_hypothesis(output_dir, hid)
        if hyp is None:
            return f"Hypothesis {hid} not found", 404

        existing = labeled_store.load_one(hid)
        if existing is not None:
            hyp = existing

        hyp.human_rating = rating
        hyp.human_notes = body.get("notes")
        hyp.labeled_at = datetime.now(UTC)
        hyp.labeled_by = "annotator"

        labeled_store.save(hyp)
        return jsonify({"ok": True})

    return app


def main():
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Browser-based hypothesis annotator")
    parser.add_argument(
        "--output-dir",
        default="geryon_data/sessions",
        help="Directory containing session JSONL files "
        "(default: geryon_data/sessions/)",
    )
    parser.add_argument(
        "--labeled-dir",
        default="geryon_data/labeled",
        help="Directory for labeled hypothesis JSON files "
        "(default: geryon_data/labeled/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to serve on (default: 8765)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labeled_store = LabeledStore(Path(args.labeled_dir))

    app = create_app(output_dir, labeled_store)

    url = f"http://localhost:{args.port}"
    print(f"Annotation server running at {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    app.run(host="localhost", port=args.port)


if __name__ == "__main__":
    main()
