"""Export labeled hypotheses to SQLite database."""

from pathlib import Path
import sqlite3

from msk_cycl.labeling.storage import HypothesisStore


def export_to_sqlite(storage_dir: Path, db_path: Path):
    """Export all labeled hypotheses to SQLite.

    Parameters
    ----------
    storage_dir : Path
        Directory containing hypothesis JSONL files
    db_path : Path
        Output SQLite database path
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hypotheses (
            hypothesis_id TEXT PRIMARY KEY,
            session_id TEXT,
            created_at TEXT,
            cohort_a_description TEXT,
            cohort_b_description TEXT,
            outcome_description TEXT,
            rationale TEXT,
            cohort_a_n INTEGER,
            cohort_b_n INTEGER,
            test_statistic TEXT,
            p_value REAL,
            summary TEXT,
            label TEXT,
            label_notes TEXT,
            labeled_at TEXT,
            labeled_by TEXT,
            llm_model TEXT
        )
    """)

    # Load all sessions
    store = HypothesisStore(storage_dir)

    # Find all JSONL files
    session_files = list(storage_dir.rglob("*.jsonl"))

    total_exported = 0
    total_skipped = 0

    for file in session_files:
        # Extract session_id from filename
        # Filename format: {session_id}_hypotheses.jsonl or similar
        session_id = file.stem.replace("_hypotheses", "")

        try:
            hypotheses = store.load_session(session_id)
        except Exception as e:
            print(f"Warning: Could not load session {session_id}: {e}")
            continue

        for hyp in hypotheses:
            # Skip pending/unlabeled
            if hyp.label.value == "pending":
                total_skipped += 1
                continue

            # Extract p-value if available
            p_value = None
            if isinstance(hyp.result.statistic, dict):
                p_value = hyp.result.statistic.get("p_value")

            cursor.execute(
                """
                INSERT OR REPLACE INTO hypotheses VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    hyp.hypothesis_id,
                    hyp.session_id,
                    (hyp.created_at.isoformat() if hyp.created_at else None),
                    hyp.proposal.cohort_a_description,
                    hyp.proposal.cohort_b_description,
                    hyp.proposal.outcome_description,
                    hyp.proposal.rationale,
                    len(hyp.result.cohort_a_data),
                    len(hyp.result.cohort_b_data),
                    str(hyp.result.statistic),
                    p_value,
                    hyp.narrative.summary,
                    hyp.label.value,
                    hyp.label_notes,
                    (hyp.labeled_at.isoformat() if hyp.labeled_at else None),
                    hyp.labeled_by,
                    hyp.llm_model,
                ),
            )

            total_exported += 1

    conn.commit()
    conn.close()

    print(f"✓ Exported {total_exported} labeled hypotheses to {db_path}")
    if total_skipped > 0:
        print(f"  Skipped {total_skipped} pending hypotheses")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Export hypotheses to SQLite")
    parser.add_argument("--storage-dir", required=True, help="Storage directory")
    parser.add_argument("--output", required=True, help="Output SQLite database path")

    args = parser.parse_args()

    export_to_sqlite(Path(args.storage_dir), Path(args.output))


if __name__ == "__main__":
    main()
