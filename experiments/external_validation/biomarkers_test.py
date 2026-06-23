"""Unit tests for biomarker extraction and altered-patient selection (no network)."""

import json

from biomarkers import STUDY_BY_CANCER_TYPE, Biomarker, extract_biomarkers
from run import _altered_patients, _os_frame


def _write_session(tmp_path, specs):
    d = tmp_path / "sessions" / "2026-01-01" / "sess"
    d.mkdir(parents=True)
    with open(d / "hypotheses.jsonl", "w") as f:
        f.write(json.dumps({"record_type": "metadata"}) + "\n")
        for s in specs:
            f.write(
                json.dumps({"record_type": "hypothesis", "data": {"spec": s}}) + "\n"
            )


def _spec(filters_a, filters_b, outcome="survival_from_treatment"):
    return {
        "query": {
            "cohort_a": {"filters": filters_a},
            "cohort_b": {"filters": filters_b},
            "outcome": {"outcome_type": outcome},
            "method": "hazard_ratio_cox",
        }
    }


def test_extracts_mutation_biomarker_with_cancer_type(tmp_path):
    spec = _spec(
        [
            {
                "table": "clinical_sample",
                "column": "CANCER_TYPE",
                "operator": "==",
                "value": "Breast Cancer",
            },
            {
                "table": "mutations_extended",
                "column": "Hugo_Symbol",
                "operator": "==",
                "value": "PIK3CA",
            },
        ],
        [
            {
                "table": "clinical_sample",
                "column": "CANCER_TYPE",
                "operator": "==",
                "value": "Breast Cancer",
            },
        ],
    )
    _write_session(tmp_path, [spec])
    bms = extract_biomarkers(str(tmp_path))
    assert bms == [
        Biomarker("Breast Cancer", "PIK3CA", "mut", 0, "brca_tcga_pan_can_atlas_2018")
    ]


def test_cna_direction_from_alt_arm_only(tmp_path):
    # value 0 is the wild-type arm and must not set direction; +2 is the alt arm.
    spec = _spec(
        [
            {
                "table": "clinical_sample",
                "column": "CANCER_TYPE",
                "operator": "==",
                "value": "Non-Small Cell Lung Cancer",
            },
            {"table": "CNA", "column": "MDM2", "operator": "==", "value": 2},
        ],
        [
            {
                "table": "clinical_sample",
                "column": "CANCER_TYPE",
                "operator": "==",
                "value": "Non-Small Cell Lung Cancer",
            },
            {"table": "CNA", "column": "MDM2", "operator": "==", "value": 0},
        ],
    )
    _write_session(tmp_path, [spec])
    bms = extract_biomarkers(str(tmp_path))
    assert len(bms) == 1
    assert bms[0].alt_kind == "cna" and bms[0].cna_dir == 1


def test_unmapped_cancer_type_skipped(tmp_path):
    spec = _spec(
        [
            {
                "table": "clinical_sample",
                "column": "CANCER_TYPE",
                "operator": "==",
                "value": "Some Rare Tumor Not In Map",
            },
            {
                "table": "mutations_extended",
                "column": "Hugo_Symbol",
                "operator": "==",
                "value": "TP53",
            },
        ],
        [],
    )
    _write_session(tmp_path, [spec])
    assert extract_biomarkers(str(tmp_path)) == []


def test_dedupes_across_hypotheses(tmp_path):
    f = [
        {
            "table": "clinical_sample",
            "column": "CANCER_TYPE",
            "operator": "==",
            "value": "Bladder Cancer",
        },
        {
            "table": "mutations_extended",
            "column": "Hugo_Symbol",
            "operator": "==",
            "value": "TP53",
        },
    ]
    _write_session(tmp_path, [_spec(f, []), _spec(f, [])])
    assert len(extract_biomarkers(str(tmp_path))) == 1


def test_label_formatting():
    assert (
        Biomarker("Breast Cancer", "TP53", "mut", 0, "x").label
        == "TP53 mut — Breast Cancer"
    )
    assert Biomarker("NSCLC", "MDM2", "cna", 1, "x").label == "MDM2 amp — NSCLC"
    assert Biomarker("NSCLC", "CDKN2A", "cna", -1, "x").label == "CDKN2A del — NSCLC"


def test_every_mapped_study_is_pancan_atlas():
    assert all(
        s.endswith("_tcga_pan_can_atlas_2018") for s in STUDY_BY_CANCER_TYPE.values()
    )


def test_altered_patients_mutation():
    bm = Biomarker("X", "KRAS", "mut", 0, "s")
    muts = {3845: {"P1", "P2"}}
    assert _altered_patients(bm, 3845, muts, {}) == {"P1", "P2"}


def test_altered_patients_cna_respects_direction():
    bm_del = Biomarker("X", "CDKN2A", "cna", -1, "s")
    bm_amp = Biomarker("X", "MDM2", "cna", 1, "s")
    cnas = {1029: {"P1": -2, "P2": 2, "P3": -2}}
    assert _altered_patients(bm_del, 1029, {}, cnas) == {"P1", "P3"}
    assert _altered_patients(bm_amp, 1029, {}, cnas) == {"P2"}
    # shallow del (-1) / gain (+1) are not "altered" — only deep events count
    assert _altered_patients(bm_del, 1029, {}, {1029: {"P9": -1}}) == set()


def test_os_frame_drops_nonpositive_time():
    os_map = {"P1": (10.0, 1), "P2": (0.0, 1), "P3": (5.0, 0)}
    df = _os_frame(os_map, {"P1", "P2", "P3"})
    assert set(df["PATIENT_ID"]) == {"P1", "P3"}
    assert set(df.columns) == {"PATIENT_ID", "time", "event"}
