"""JSONL storage for labeled hypotheses."""

from datetime import datetime
import json
from pathlib import Path

import pandas as pd  # type: ignore

from msk_cycl.labeling.labels import HypothesisLabel
from msk_cycl.labeling.models import LabeledHypothesis
from msk_cycl.labeling.schema import HypothesisRecord, SessionFileMetadata


class HypothesisStore:
    """JSONL storage for labeled hypotheses with schema validation."""

    def __init__(self, storage_dir: Path | str):
        """Initialize storage.

        Parameters
        ----------
        storage_dir : Path | str
            Directory for JSONL session files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def initialize_session(self, session_id: str) -> None:
        """Initialize session file with metadata record.

        Parameters
        ----------
        session_id : str
            Session identifier
        """
        session_file = self.storage_dir / f"{session_id}.jsonl"

        if session_file.exists():
            return  # Already initialized

        metadata = SessionFileMetadata(
            session_id=session_id,
            created_at=datetime.utcnow(),
        )

        with open(session_file, "w") as f:
            f.write(metadata.model_dump_json() + "\n")

    def save(self, hypothesis: LabeledHypothesis) -> None:
        """Append hypothesis to session JSONL file.

        Parameters
        ----------
        hypothesis : LabeledHypothesis
            Hypothesis to save
        """
        session_file = self.storage_dir / f"{hypothesis.session_id}.jsonl"

        # Ensure session file exists with metadata
        if not session_file.exists():
            self.initialize_session(hypothesis.session_id)

        # Create hypothesis record
        record = HypothesisRecord(data=hypothesis)

        # Serialize with custom encoder for pandas DataFrames
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

                # Skip metadata line
                if data.get("record_type") == "metadata":
                    continue

                # Parse hypothesis record
                record = self._deserialize_record(data)
                hypotheses.append(record.data)

        return hypotheses

    def load_all(self) -> list[LabeledHypothesis]:
        """Load all hypotheses across all sessions.

        Returns
        -------
        list[LabeledHypothesis]
            All hypotheses
        """
        all_hypotheses = []
        for session_file in self.storage_dir.glob("*.jsonl"):
            session_id = session_file.stem
            hypotheses = self.load_session(session_id)
            all_hypotheses.extend(hypotheses)
        return all_hypotheses

    def update_label(
        self,
        hypothesis_id: str,
        session_id: str,
        label: HypothesisLabel,
        notes: str | None = None,
        labeled_by: str | None = None,
    ) -> None:
        """Update label for a hypothesis (rewrites session file).

        Parameters
        ----------
        hypothesis_id : str
            Hypothesis identifier
        session_id : str
            Session identifier
        label : HypothesisLabel
            New label
        notes : str, optional
            Reviewer notes
        labeled_by : str, optional
            Reviewer identifier
        """
        hypotheses = self.load_session(session_id)

        # Find and update hypothesis
        found = False
        for hyp in hypotheses:
            if hyp.hypothesis_id == hypothesis_id:
                hyp.label = label
                hyp.label_notes = notes
                hyp.labeled_by = labeled_by
                hyp.labeled_at = datetime.utcnow()
                found = True
                break

        if not found:
            raise ValueError(
                f"Hypothesis {hypothesis_id} not found in session {session_id}"
            )

        # Rewrite session file
        session_file = self.storage_dir / f"{session_id}.jsonl"

        # Write metadata
        metadata = SessionFileMetadata(
            session_id=session_id,
            created_at=hypotheses[0].created_at if hypotheses else datetime.utcnow(),
        )

        with open(session_file, "w") as f:
            f.write(metadata.model_dump_json() + "\n")

            # Write updated hypotheses
            for hyp in hypotheses:
                record = HypothesisRecord(data=hyp)
                json_line = self._serialize_record(record) + "\n"
                f.write(json_line)

    def _serialize_record(self, record: HypothesisRecord) -> str:
        """Serialize record with custom DataFrame handling.

        Parameters
        ----------
        record : HypothesisRecord
            Record to serialize

        Returns
        -------
        str
            JSON string
        """
        # Convert to dict with custom DataFrame serialization
        data = record.model_dump()

        # Convert DataFrames in result to dict
        if "data" in data and "result" in data["data"]:
            result = data["data"]["result"]
            if "cohort_a_data" in result:
                result["cohort_a_data"] = result["cohort_a_data"].to_dict(
                    orient="records"
                )
            if "cohort_b_data" in result:
                result["cohort_b_data"] = result["cohort_b_data"].to_dict(
                    orient="records"
                )

        return json.dumps(data, default=str)

    def _deserialize_record(self, data: dict) -> HypothesisRecord:
        """Deserialize record with custom DataFrame handling.

        Parameters
        ----------
        data : dict
            JSON data

        Returns
        -------
        HypothesisRecord
            Deserialized record
        """
        # Convert dict DataFrames back to pandas
        if "data" in data and "result" in data["data"]:
            result = data["data"]["result"]
            if "cohort_a_data" in result and isinstance(result["cohort_a_data"], list):
                result["cohort_a_data"] = pd.DataFrame(result["cohort_a_data"])
            if "cohort_b_data" in result and isinstance(result["cohort_b_data"], list):
                result["cohort_b_data"] = pd.DataFrame(result["cohort_b_data"])

        return HypothesisRecord(**data)
