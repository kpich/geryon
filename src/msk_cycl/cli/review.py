"""Interactive CLI for reviewing and labeling hypotheses."""

from datetime import UTC, datetime
from pathlib import Path

from msk_cycl.labeling.labels import HypothesisRating
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
        pending = [h for h in hypotheses if h.rating.is_pending]

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
            f"  N_A = {hyp.result.cohort_a_size}, " f"N_B = {hyp.result.cohort_b_size}"
        )
        if hyp.result.p_value is not None:
            print(f"  p-value = {hyp.result.p_value}")
        if hyp.result.hazard_ratio is not None:
            print(f"  hazard_ratio = {hyp.result.hazard_ratio}")
            if hyp.result.confidence_interval_lower is not None:
                ci_lower = hyp.result.confidence_interval_lower
                ci_upper = hyp.result.confidence_interval_upper
                print(f"  95% CI = [{ci_lower}, {ci_upper}]")
        print()
        print(f"Summary: {hyp.narrative.summary}")
        print()

        novelty = self._prompt_dimension("Novelty (1-3, blank to skip): ")
        uncontrolled = self._prompt_dimension("Uncontrolled (1-3, blank to skip): ")
        trustworthiness = self._prompt_dimension(
            "Trustworthiness (1-3, blank to skip): "
        )
        is_duplicate = self._prompt_bool("Duplicate? (y/n, blank to skip): ")

        rating = HypothesisRating(
            novelty=novelty,
            uncontrolled=uncontrolled,
            trustworthiness=trustworthiness,
            is_duplicate=is_duplicate,
        )

        if rating.is_pending:
            print("Skipped (no dimensions rated)")
            print()
            return

        notes = input("Notes (optional): ").strip()

        hyp.rating = rating
        hyp.notes = notes if notes else None
        hyp.labeled_at = datetime.now(UTC)
        hyp.labeled_by = reviewer

        self.store.save(hyp)

        print("Saved")
        print()

    @staticmethod
    def _prompt_dimension(prompt: str) -> int | None:
        while True:
            val = input(prompt).strip()
            if not val:
                return None
            if val in ("1", "2", "3"):
                return int(val)
            print("Enter 1, 2, or 3 (or blank to skip).")

    @staticmethod
    def _prompt_bool(prompt: str) -> bool | None:
        while True:
            val = input(prompt).strip().lower()
            if not val:
                return None
            if val in ("y", "yes"):
                return True
            if val in ("n", "no"):
                return False
            print("Enter y/n (or blank to skip).")


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
