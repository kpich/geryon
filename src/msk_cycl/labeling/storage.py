"""JSONL storage for labeled hypotheses."""

from datetime import datetime
import json
from pathlib import Path

from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.schema import HypothesisRecord, SessionFileMetadata

SUPPORTED_VERSIONS = {1}


class HypothesisStore:
    """JSONL storage for labeled hypotheses.

    Simple append-only interface: save() creates session files as needed,
    load_session() reads all hypotheses from a session.
    """

    def __init__(self, storage_dir: Path | str):
        """Initialize storage.

        Parameters
        ----------
        storage_dir : Path | str
            Directory for JSONL session files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, hypothesis: LabeledHypothesis) -> None:
        """Append hypothesis to session file (creates if needed).

        Parameters
        ----------
        hypothesis : LabeledHypothesis
            Hypothesis to save
        """
        session_file = self.storage_dir / f"{hypothesis.session_id}.jsonl"

        if not session_file.exists():
            metadata = SessionFileMetadata(
                session_id=hypothesis.session_id,
                created_at=datetime.utcnow(),
            )
            with open(session_file, "w") as f:
                f.write(metadata.model_dump_json() + "\n")

        record = HypothesisRecord(data=hypothesis)

        json_line = self._serialize_record(record) + "\n"

        with open(session_file, "a") as f:
            f.write(json_line)

    def load_session(self, session_id: str) -> list[LabeledHypothesis]:
        """Load all hypotheses from a session.

        Parameters
        ----------
        session_id : str
            Session identifier

        Returns
        -------
        list[LabeledHypothesis]
            Hypotheses in the session
        """
        session_file = self.storage_dir / f"{session_id}.jsonl"

        if not session_file.exists():
            return []

        hypotheses = []
        with open(session_file) as f:
            for line in f:
                data = json.loads(line)

                if data.get("record_type") == "metadata":
                    continue

                record = self._deserialize_record(data)
                hypotheses.append(record.data)

        return hypotheses

    def _serialize_record(self, record: HypothesisRecord) -> str:
        """Serialize record, dropping bulk fields that are recomputable from spec."""
        data = record.model_dump()

        if "data" in data and "result" in data["data"]:
            result = data["data"]["result"]
            result.pop("cohort_a_data", None)
            result.pop("cohort_b_data", None)

        return json.dumps(data, default=str)

    def _deserialize_record(self, data: dict) -> HypothesisRecord:
        """Deserialize record, validating spec version."""
        if "data" in data and "spec" in data["data"]:
            spec = data["data"]["spec"]
            version = spec.get("version")
            if version not in SUPPORTED_VERSIONS:
                raise ValueError(f"Unsupported CyclHyp version: {version}")

        return HypothesisRecord(**data)
