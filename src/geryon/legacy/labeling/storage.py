"""JSONL storage for labeled hypotheses."""

from datetime import UTC, datetime
import json
from pathlib import Path

from geryon.legacy.labeling.models import LabeledHypothesis
from geryon.legacy.labeling.schema import HypothesisRecord, SessionFileMetadata

SUPPORTED_VERSIONS = {1}
HYPOTHESES_FILENAME = "hypotheses.jsonl"


class HypothesisStore:
    """JSONL storage for labeled hypotheses.

    Simple append-only interface: save() appends to the session file,
    load() reads all hypotheses back. The directory _is_ the session.
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
        session_file = self.storage_dir / HYPOTHESES_FILENAME

        if not session_file.exists():
            metadata = SessionFileMetadata(
                session_id=hypothesis.session_id,
                created_at=datetime.now(UTC),
            )
            with open(session_file, "w") as f:
                f.write(metadata.model_dump_json() + "\n")

        record = HypothesisRecord(data=hypothesis)

        json_line = self._serialize_record(record) + "\n"

        with open(session_file, "a") as f:
            f.write(json_line)

    def save_all(self, hypotheses: list[LabeledHypothesis]) -> None:
        """Rewrite session file with updated hypotheses (atomic via temp+rename)."""
        if not hypotheses:
            return

        session_file = self.storage_dir / HYPOTHESES_FILENAME
        tmp_file = session_file.with_suffix(".jsonl.tmp")

        metadata_line = None
        if session_file.exists():
            with open(session_file) as f:
                first = json.loads(f.readline())
                if first.get("record_type") == "metadata":
                    metadata_line = json.dumps(first, default=str)

        if metadata_line is None:
            meta = SessionFileMetadata(
                session_id=hypotheses[0].session_id,
                created_at=datetime.now(UTC),
            )
            metadata_line = meta.model_dump_json()

        with open(tmp_file, "w") as f:
            f.write(metadata_line + "\n")
            for hyp in hypotheses:
                record = HypothesisRecord(data=hyp)
                f.write(self._serialize_record(record) + "\n")

        tmp_file.rename(session_file)

    def load(self) -> list[LabeledHypothesis]:
        """Load all hypotheses from this store's directory.

        Returns
        -------
        list[LabeledHypothesis]
            Hypotheses in the session
        """
        session_file = self.storage_dir / HYPOTHESES_FILENAME

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
                raise ValueError(f"Unsupported GeryonHyp version: {version}")

        return HypothesisRecord(**data)
