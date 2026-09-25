"""Tests for the in-container runtime helpers.

runtime.py is standalone (no geryon imports) so it can run inside the image, but it
imports cleanly on the host too, so we exercise it here by pointing its module-level
paths at a tmp dir. The JSON it writes must round-trip through IterationResult, which
is the contract the host runner relies on.
"""

import json
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest

from geryon.sandbox import runtime
from geryon.sandbox.result import IterationResult


def _point_scratch(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(runtime, "SCRATCH_DIR", tmp_path)
    monkeypatch.setattr(runtime, "RESULT_PATH", tmp_path / "result.json")
    return tmp_path / "result.json"


def test_report_writes_result_that_validates(monkeypatch, tmp_path: Path):
    result_path = _point_scratch(monkeypatch, tmp_path)

    runtime.report(
        effect_size=1.8,
        effect_size_type="hazard_ratio",
        p_value=0.003,
        ci=(1.2, 2.7),
        n_a=200,
        n_b=180,
        summary="TP53-mut worse OS",
        extra={"fdr": 0.01},
    )

    data = json.loads(result_path.read_text())
    parsed = IterationResult.model_validate(data)
    assert parsed.effect_size == 1.8
    assert parsed.ci_lower == 1.2
    assert parsed.ci_upper == 2.7
    assert parsed.n_a == 200
    assert parsed.extra == {"fdr": 0.01}


def test_report_partial_is_allowed(monkeypatch, tmp_path: Path):
    result_path = _point_scratch(monkeypatch, tmp_path)

    runtime.report(summary="no clean effect size")

    parsed = IterationResult.model_validate(json.loads(result_path.read_text()))
    assert parsed.summary == "no clean effect size"
    assert parsed.effect_size is None
    assert parsed.extra == {}


def test_report_last_write_wins(monkeypatch, tmp_path: Path):
    result_path = _point_scratch(monkeypatch, tmp_path)

    runtime.report(effect_size=1.0)
    runtime.report(effect_size=2.0)

    parsed = IterationResult.model_validate(json.loads(result_path.read_text()))
    assert parsed.effect_size == 2.0


def test_db_registers_parquet_views(monkeypatch, tmp_path: Path):
    df = pd.DataFrame({"PATIENT_ID": ["P1", "P2"], "OS_MONTHS": [10.0, 20.0]})
    df.to_parquet(tmp_path / "data_clinical_patient.parquet", index=False)
    monkeypatch.setattr(runtime, "DATA_DIR", tmp_path)

    con = runtime.db()
    out = con.execute("SELECT COUNT(*) FROM clinical_patient").fetchone()
    assert out is not None
    assert out[0] == 2


def test_table_name_strips_data_prefix():
    assert runtime._table_name(Path("data_CNA.parquet")) == "CNA"
    timeline = Path("timeline_treatment.parquet")
    assert runtime._table_name(timeline) == "timeline_treatment"


def test_report_extra_accepts_non_numeric_values(monkeypatch, tmp_path: Path):
    result_path = _point_scratch(monkeypatch, tmp_path)

    extra = {
        "censoring": "first T-DXd dose; censored at global last follow-up",
        "hr_by_line": {"1L": 0.8, "2L+": 1.3},
        "n_events": np.int64(42),
        "median_os": np.float64(18.5),
        "cutpoints": np.array([6, 12, 24]),
        "missing": None,
    }
    runtime.report(extra=extra)

    parsed = IterationResult.model_validate(json.loads(result_path.read_text()))
    assert parsed.extra == {
        "censoring": "first T-DXd dose; censored at global last follow-up",
        "hr_by_line": {"1L": 0.8, "2L+": 1.3},
        "n_events": 42,
        "median_os": 18.5,
        "cutpoints": [6, 12, 24],
        "missing": None,
    }


def test_report_extra_rejects_unserializable_value(monkeypatch, tmp_path: Path):
    _point_scratch(monkeypatch, tmp_path)

    with pytest.raises(TypeError, match="JSON-serializable"):
        runtime.report(extra={"obj": object()})
