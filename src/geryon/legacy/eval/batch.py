"""Batch hypothesis re-execution against a configured executor."""

import argparse
import csv
from pathlib import Path

from geryon.db import Database
from geryon.legacy.engine import HypothesisExecutor, load_split_ids
from geryon.legacy.labeling.models import LabeledHypothesis
from geryon.legacy.plot._loader import load_hypotheses
from geryon.legacy.tools.derived import replay_derived_views


def execute_hypotheses(
    hypotheses: list[LabeledHypothesis],
    executor: HypothesisExecutor,
) -> list[dict]:
    """Re-execute hypothesis specs and return per-hypothesis result dicts.

    Each dict contains: hypothesis_id, p_value, cohort_a_size, cohort_b_size,
    success, error_message.
    """
    results = []
    for hyp in hypotheses:
        result = executor.execute(hyp.spec)
        results.append(
            {
                "hypothesis_id": hyp.hypothesis_id,
                "p_value": result.p_value,
                "cohort_a_size": result.cohort_a_size,
                "cohort_b_size": result.cohort_b_size,
                "success": result.success,
                "error_message": result.error_message,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-execute hypothesis specs against the validation cohort."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Geryon data dir (contains sessions/)",
    )
    parser.add_argument(
        "--parquet-dir", required=True, type=Path, help="Parquet data dir (from ETL)"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    hypotheses = load_hypotheses(args.data_dir)
    train_p_by_id = {hyp.hypothesis_id: hyp.result.p_value for hyp in hypotheses}

    with Database(args.parquet_dir) as db:
        # Hypotheses reference session-local derived_* views; recreate them in the
        # fresh val DB or every such hypothesis fails with a Catalog Error.
        for views_path in sorted(
            (args.data_dir / "sessions").rglob("derived_views.json")
        ):
            for name in replay_derived_views(db, views_path):
                print(f"  ⚠ Could not replay derived view '{name}' from {views_path}")

        val_ids = load_split_ids(db, "validation")
        if val_ids is None:
            raise SystemExit("No patient_split table found in parquet-dir.")
        executor = HypothesisExecutor(db, patient_ids=val_ids)
        val_results = execute_hypotheses(hypotheses, executor)

    fieldnames = [
        "hypothesis_id",
        "train_p_value",
        "val_p_value",
        "val_cohort_a_size",
        "val_cohort_b_size",
        "val_success",
        "val_error_message",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in val_results:
            writer.writerow(
                {
                    "hypothesis_id": row["hypothesis_id"],
                    "train_p_value": train_p_by_id.get(row["hypothesis_id"]),
                    "val_p_value": row["p_value"],
                    "val_cohort_a_size": row["cohort_a_size"],
                    "val_cohort_b_size": row["cohort_b_size"],
                    "val_success": row["success"],
                    "val_error_message": row["error_message"],
                }
            )


if __name__ == "__main__":
    main()
