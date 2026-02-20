"""Shared hypothesis loading for plot scripts."""

from pathlib import Path

from geryon.labeling.labeled_store import LabeledStore
from geryon.labeling.models import LabeledHypothesis
from geryon.labeling.storage import HypothesisStore


def load_hypotheses(data_dir: Path) -> list[LabeledHypothesis]:
    """Load all hypotheses from sessions, merging human labels from labeled store."""
    lstore = LabeledStore(data_dir / "labeled")
    seen: dict[str, LabeledHypothesis] = {}
    for jsonl_file in sorted((data_dir / "sessions").rglob("hypotheses.jsonl")):
        store = HypothesisStore(jsonl_file.parent)
        try:
            hyps = store.load()
        except Exception:
            continue
        for hyp in hyps:
            if hyp.hypothesis_id in seen:
                continue
            labeled = lstore.load_one(hyp.hypothesis_id)
            if labeled is not None and labeled.human_rating is not None:
                hyp = hyp.model_copy(update={"human_rating": labeled.human_rating})
            seen[hyp.hypothesis_id] = hyp
    return sorted(seen.values(), key=lambda h: h.created_at)
