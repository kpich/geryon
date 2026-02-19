"""Per-file storage for labeled hypotheses.

Each labeled hypothesis is stored as a separate JSON file in a directory,
named by hypothesis_id. This avoids append-corruption issues with JSONL
and makes it easy to glob/read for the LLM loop.
"""

import json
from pathlib import Path

from geryon.labeling.models import LabeledHypothesis


class LabeledStore:
    """Read/write individual JSON files in a labeled-hypotheses directory."""

    def __init__(self, labeled_dir: Path | str):
        self.labeled_dir = Path(labeled_dir)
        self.labeled_dir.mkdir(parents=True, exist_ok=True)

    def save(self, hypothesis: LabeledHypothesis) -> None:
        """Write one JSON file per labeled hypothesis."""
        data = hypothesis.model_dump()

        if "result" in data:
            data["result"].pop("cohort_a_data", None)
            data["result"].pop("cohort_b_data", None)

        path = self.labeled_dir / f"{hypothesis.hypothesis_id}.json"
        path.write_text(json.dumps(data, default=str, indent=2))

    def load_all(self) -> list[LabeledHypothesis]:
        """Read all labeled hypotheses from directory."""
        results = []
        for path in sorted(self.labeled_dir.glob("*.json")):
            data = json.loads(path.read_text())
            results.append(LabeledHypothesis(**data))
        return results

    def load_one(self, hypothesis_id: str) -> LabeledHypothesis | None:
        """Load a single labeled hypothesis by ID, or None if not found."""
        path = self.labeled_dir / f"{hypothesis_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return LabeledHypothesis(**data)

    def labeled_ids(self) -> set[str]:
        """Return set of already-labeled hypothesis_ids (fast, filenames only)."""
        return {p.stem for p in self.labeled_dir.glob("*.json")}
