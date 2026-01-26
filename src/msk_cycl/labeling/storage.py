"""JSONL storage for labeled hypotheses."""

import contextlib
from datetime import datetime
import json
from pathlib import Path

import pandas as pd  # type: ignore

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
        data = record.model_dump()

        if "data" in data and "result" in data["data"]:
            result = data["data"]["result"]
            if "cohort_a_data" in result:
                df = result["cohort_a_data"]
                result["cohort_a_data"] = {
                    "records": df.to_dict(orient="records"),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }
            if "cohort_b_data" in result:
                df = result["cohort_b_data"]
                result["cohort_b_data"] = {
                    "records": df.to_dict(orient="records"),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }

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
        if "data" in data and "spec" in data["data"]:
            spec = data["data"]["spec"]
            version = spec.get("version")
            if version not in SUPPORTED_VERSIONS:
                raise ValueError(f"Unsupported CyclHyp version: {version}")

        if "data" in data and "result" in data["data"]:
            result = data["data"]["result"]
            for key in ("cohort_a_data", "cohort_b_data"):
                if key in result:
                    cohort_data = result[key]
                    if isinstance(cohort_data, list):
                        result[key] = pd.DataFrame(cohort_data)
                    elif isinstance(cohort_data, dict) and "records" in cohort_data:
                        df = pd.DataFrame(cohort_data["records"])
                        if "dtypes" in cohort_data:
                            for col, dtype_str in cohort_data["dtypes"].items():
                                if col in df.columns:
                                    with contextlib.suppress(ValueError, TypeError):
                                        df[col] = df[col].astype(dtype_str)
                        result[key] = df

        return HypothesisRecord(**data)
