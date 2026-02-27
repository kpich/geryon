"""Generate a self-contained HTML report of all hypotheses and ratings.

Embeds all data as JSON in a <script> tag — no server required.
"""

from datetime import datetime
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from geryon.cli.annotate import _load_all
from geryon.labeling.labeled_store import LabeledStore
from geryon.labeling.labels import RATING_DIMENSIONS


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML hypothesis report"
    )
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
        "--out",
        default=None,
        help="Output file path "
        "(default: hypothesis_report_<YYYYMMDD_HHMMSS>.html in CWD)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    labeled_store = LabeledStore(Path(args.labeled_dir))

    hypotheses = _load_all(output_dir, labeled_store)

    total = len(hypotheses)
    human_labeled = sum(1 for h in hypotheses if h["status"] == "human_labeled")
    critic_labeled = sum(1 for h in hypotheses if h["status"] == "critic_labeled")
    pending = sum(1 for h in hypotheses if h["status"] == "pending")
    stats = {
        "total": total,
        "human_labeled": human_labeled,
        "critic_labeled": critic_labeled,
        "pending": pending,
    }

    dimensions = {
        k: {
            "label": v["label"],
            "levels": {str(lk): lv for lk, lv in v["levels"].items()},
        }
        for k, v in RATING_DIMENSIONS.items()
    }

    hypotheses_json = json.dumps(hypotheses)
    dimensions_json = json.dumps(dimensions)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(FileSystemLoader(str(templates_dir)), autoescape=False)
    template = env.get_template("report.html")

    rendered = template.render(
        hypotheses_json=hypotheses_json,
        dimensions_json=dimensions_json,
        stats=stats,
        generated_at=generated_at,
    )

    if args.out:
        out_path = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"hypothesis_report_{timestamp}.html")

    out_path.write_text(rendered, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
