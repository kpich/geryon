"""Browser-based viewer for code-first hypotheses.

Read-only Flask app that surfaces ``CodeHypothesis`` records written by the
codeflow runner under ``geryon_data/sessions/``. The legacy annotator
(``geryon.legacy.cli.annotate``) was bound to the old spec model; this is its
replacement for the overhaul.

Kept deliberately thin: the API just serves the stored records as JSON and the
template renders them. New fields on ``CodeHypothesis`` show up with no server
change, and richer visualizations (plots, lineage graphs) can be layered on the
client over the same ``/api/hypotheses`` payload.
"""

import argparse
from pathlib import Path
import webbrowser

from flask import Flask, jsonify, render_template

from geryon.codeflow.models import CodeHypothesis
from geryon.codeflow.store import CodeHypothesisStore


def _load_all(output_dir: Path) -> list[dict]:
    """Load every CodeHypothesis under output_dir, newest session last.

    Sorted by (created_at, iteration) so lineage reads top-to-bottom within a
    session. Each record carries its session_dir so the UI can group / link to
    the on-disk artifacts (trace.jsonl, detail.jsonl).
    """
    out: list[dict] = []
    for jsonl_file in sorted(output_dir.rglob("hypotheses.jsonl")):
        store = CodeHypothesisStore(jsonl_file.parent)
        try:
            hypotheses = store.load()
        except Exception:
            continue
        for hyp in hypotheses:
            out.append(_serialize(hyp, jsonl_file.parent))

    out.sort(key=lambda h: (h["created_at"], h["iteration"] or 0))
    return out


def _serialize(hyp: CodeHypothesis, session_dir: Path) -> dict:
    """Flatten a CodeHypothesis to a JSON-friendly dict for the template."""
    d = hyp.model_dump(mode="json")
    d["short_id"] = hyp.short_id()
    d["refines_short"] = hyp.refines[:8] if hyp.refines else None
    d["session_dir"] = str(session_dir)
    return d


def _stats(hyps: list[dict]) -> dict:
    sessions = {h["session_id"] for h in hyps}
    return {
        "total": len(hyps),
        "sessions": len(sessions),
        "succeeded": sum(1 for h in hyps if h["success"]),
        "failed": sum(1 for h in hyps if not h["success"]),
    }


def create_app(output_dir: Path) -> Flask:
    """App factory for the viewer."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("viewer.html")

    @app.get("/api/hypotheses")
    def api_hypotheses():
        # Hide crashed/timed-out runs (success=False): a failed submit that was
        # fixed and resubmitted moments later is noise. They remain on disk and
        # are still counted in /api/stats.
        return jsonify([h for h in _load_all(output_dir) if h["success"]])

    @app.get("/api/stats")
    def api_stats():
        return jsonify(_stats(_load_all(output_dir)))

    return app


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Code-first hypothesis viewer")
    parser.add_argument(
        "--output-dir",
        default="geryon_data/sessions",
        help="Directory of code session JSONL files (default: geryon_data/sessions/)",
    )
    parser.add_argument(
        "--port", type=int, default=8765, help="Port to serve on (default: 8765)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open a browser"
    )
    args = parser.parse_args()

    app = create_app(Path(args.output_dir))

    url = f"http://localhost:{args.port}"
    print(f"Viewer running at {url}")
    print("Press Ctrl+C to stop.\n")
    if not args.no_browser:
        webbrowser.open(url)
    app.run(host="localhost", port=args.port)


if __name__ == "__main__":
    main()
