"""Interactive CLI for reviewing and labeling hypotheses."""

from datetime import UTC, datetime
from pathlib import Path

from msk_cycl.labeling.labels import HypothesisLabel
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.storage import HypothesisStore


class HypothesisReviewer:
    """Interactive hypothesis review interface."""

    def __init__(self, storage_dir: Path):
        """Initialize reviewer.

        Parameters
        ----------
        storage_dir : Path
            Directory containing hypothesis storage files
        """
        self.store = HypothesisStore(storage_dir)

    def review_session(self, session_id: str, reviewer_name: str):
        """Review all hypotheses in a session.

        Parameters
        ----------
        session_id : str
            Session ID to review
        reviewer_name : str
            Name/ID of the reviewer
        """
        hypotheses = self.store.load_session(session_id)
        pending = [h for h in hypotheses if h.label == HypothesisLabel.PENDING]

        print(f"Found {len(pending)} pending hypotheses in session {session_id}")
        print()

        for i, hyp in enumerate(pending, 1):
            self._review_one(hyp, i, len(pending), reviewer_name)

        print("=" * 80)
        print("Review complete!")
        print()

    def _review_one(self, hyp: LabeledHypothesis, idx: int, total: int, reviewer: str):
        """Review a single hypothesis.

        Parameters
        ----------
        hyp : LabeledHypothesis
            Hypothesis to review
        idx : int
            Current index (1-based)
        total : int
            Total number of hypotheses
        reviewer : str
            Reviewer name/ID
        """
        print("=" * 80)
        print(f"Hypothesis {idx}/{total}")
        print("=" * 80)
        print()
        print(f"Cohort A: {hyp.proposal.cohort_a_description}")
        print(f"Cohort B: {hyp.proposal.cohort_b_description}")
        print(f"Outcome:  {hyp.proposal.outcome_description}")
        print()
        print(f"Rationale: {hyp.proposal.rationale}")
        print()
        print("Results:")
        print(
            f"  N_A = {len(hyp.result.cohort_a_data)}, "
            f"N_B = {len(hyp.result.cohort_b_data)}"
        )
        if isinstance(hyp.result.statistic, dict):
            print(f"  p-value = {hyp.result.statistic.get('p_value', 'N/A')}")
            test_stat = hyp.result.statistic.get("test_statistic", "N/A")
            print(f"  test_statistic = {test_stat}")
        else:
            print(f"  Statistic: {hyp.result.statistic}")
        print()
        print(f"Summary: {hyp.narrative.summary}")
        print()

        # Show label options
        print("Labels:")
        for label in HypothesisLabel:
            if label != HypothesisLabel.PENDING:
                print(f"  {label.value}: {label.name}")
        print()

        # Get user input
        while True:
            label_input = (
                input(
                    "Label [correct/red_herring/confounded/data_issue/"
                    "duplicate/not_novel/skip]: "
                )
                .strip()
                .lower()
            )

            if label_input == "skip":
                print("Skipped")
                print()
                return

            try:
                label = HypothesisLabel(label_input)
                break
            except ValueError:
                print(f"Invalid label: {label_input}. Try again.")

        notes = input("Notes (optional): ").strip()

        # Update hypothesis
        hyp.label = label
        hyp.label_notes = notes if notes else None
        hyp.labeled_at = datetime.now(UTC)
        hyp.labeled_by = reviewer

        # Save updated hypothesis
        self.store.save(hyp)

        print("✓ Saved")
        print()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Review and label hypotheses")
    parser.add_argument("--storage-dir", required=True, help="Storage directory")
    parser.add_argument("--session", required=True, help="Session ID to review")
    parser.add_argument("--reviewer", required=True, help="Your name/ID")

    args = parser.parse_args()

    reviewer = HypothesisReviewer(Path(args.storage_dir))
    reviewer.review_session(args.session, args.reviewer)


if __name__ == "__main__":
    main()
