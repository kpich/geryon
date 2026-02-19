"""Browser-based annotation server for labeling hypotheses.

Flask app that serves a template and JSON API. Lets reviewers rate
hypotheses on multiple dimensions instead of typing label strings
at a terminal prompt.
"""

from datetime import UTC, datetime
from pathlib import Path
import webbrowser

from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError

from geryon.labeling.labeled_store import LabeledStore
from geryon.labeling.labels import RATING_DIMENSIONS, HypothesisRating
from geryon.labeling.storage import HypothesisStore


def _load_unlabeled(output_dir: Path, labeled_store: LabeledStore) -> list[dict]:
    """Load all hypotheses from session JSONLs, filter to unlabeled, newest first."""
    labeled_ids = labeled_store.labeled_ids()

    seen: dict[str, dict] = {}
    for jsonl_file in sorted(output_dir.rglob("hypotheses.jsonl")):
        store = HypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load()
        except Exception:
            continue

        for hyp in hypotheses:
            hid = hyp.hypothesis_id
            if hid in labeled_ids or hid in seen:
                continue
            if not hyp.rating.is_pending and hyp.labeled_by != "llm_critic":
                continue

            seen[hid] = {
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
                "spec": hyp.spec.model_dump(),
                "summary": hyp.narrative.summary,
                "findings": hyp.narrative.findings,
                "limitations": hyp.narrative.limitations,
                "clinical_relevance": hyp.narrative.clinical_relevance,
                "iteration": hyp.iteration,
                "labeled_by": hyp.labeled_by,
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
            }

    items = list(seen.values())
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def _count_all(output_dir: Path, labeled_store: LabeledStore) -> dict:
    """Return total / labeled / pending counts."""
    labeled_ids = labeled_store.labeled_ids()

    all_ids: set[str] = set()
    for jsonl_file in sorted(output_dir.rglob("hypotheses.jsonl")):
        store = HypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load()
        except Exception:
            continue
        for hyp in hypotheses:
            all_ids.add(hyp.hypothesis_id)

    total = len(all_ids)
    labeled = len(labeled_ids & all_ids)
    return {"total": total, "labeled": labeled, "pending": total - labeled}


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
        return jsonify(_load_unlabeled(output_dir, labeled_store))

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

        hyp.rating = rating
        hyp.notes = body.get("notes")
        hyp.labeled_at = datetime.now(UTC)
        hyp.labeled_by = "annotator"

        labeled_store.save(hyp)
        return jsonify({"ok": True})

    return app


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Browser-based hypothesis annotator")
    parser.add_argument(
        "--output-dir",
        default="geryon_run_outputs",
        help="Directory containing session JSONL files (default: geryon_run_outputs/)",
    )
    parser.add_argument(
        "--labeled-dir",
        default="labeled_hypotheses",
        help="Directory for labeled hypothesis JSON files "
        "(default: labeled_hypotheses/)",
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
