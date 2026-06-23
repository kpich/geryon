"""Extract portable genomic biomarkers from the Geryon hypothesis corpus.

The corpus is entirely treatment-anchored (survival_from_treatment), which does
not port to TCGA. But each hypothesis is built on a genomic stratifier
(gene x alteration x cancer type). We extract that stratifier so we can ask the
relaxed, independent question on TCGA: does the biomarker associate with OS?

This is a deliberate relaxation: it drops the treatment context. See README.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os

# MSK CANCER_TYPE -> TCGA PanCancer Atlas study. Single best-matched study per
# type (NSCLC -> LUAD adenocarcinoma, where KRAS/EGFR/STK11/KEAP1 concentrate;
# RCC -> KIRC clear cell, the VHL/PBRM1/SETD2/BAP1 context).
STUDY_BY_CANCER_TYPE: dict[str, str] = {
    "Bladder Cancer": "blca_tcga_pan_can_atlas_2018",
    "Breast Cancer": "brca_tcga_pan_can_atlas_2018",
    "Endometrial Cancer": "ucec_tcga_pan_can_atlas_2018",
    "Esophagogastric Cancer": "stad_tcga_pan_can_atlas_2018",
    "Non-Small Cell Lung Cancer": "luad_tcga_pan_can_atlas_2018",
    "Pancreatic Cancer": "paad_tcga_pan_can_atlas_2018",
    "Prostate Cancer": "prad_tcga_pan_can_atlas_2018",
    "Renal Cell Carcinoma": "kirc_tcga_pan_can_atlas_2018",
}


@dataclass(frozen=True)
class Biomarker:
    """A gene x alteration x cancer-type contrast to test against TCGA OS."""

    cancer_type: str
    gene: str
    alt_kind: str  # "mut" or "cna"
    cna_dir: int  # +1 (amplification), -1 (deletion), 0 (n/a for mutations)
    study_id: str

    @property
    def label(self) -> str:
        if self.alt_kind == "mut":
            alt = "mut"
        else:
            alt = "amp" if self.cna_dir > 0 else "del"
        return f"{self.gene} {alt} — {self.cancer_type}"


def _iter_specs(data_dir: str):
    pattern = os.path.join(data_dir, "sessions", "*", "*", "hypotheses.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("record_type") != "hypothesis":
                    continue
                spec = rec["data"].get("spec")
                if spec:
                    yield spec


def extract_biomarkers(data_dir: str) -> list[Biomarker]:
    """Extract the deduplicated set of mappable biomarkers from the corpus."""
    seen: set[Biomarker] = set()
    for spec in _iter_specs(data_dir):
        q = spec["query"]
        filters = q["cohort_a"]["filters"] + q["cohort_b"]["filters"]
        cancer_types = {
            f["value"]
            for f in filters
            if f["column"] in ("CANCER_TYPE", "CANCER_TYPE_DETAILED")
            and isinstance(f["value"], str)
        }
        muts = {
            f["value"]
            for f in filters
            if f["table"] == "mutations_extended" and f["column"] == "Hugo_Symbol"
        }
        # CNA filters carry the gene in `column` and the GISTIC level in `value`.
        cnas: dict[str, int] = {}
        for f in filters:
            if f["table"] == "CNA" and isinstance(f["value"], int | float):
                val = int(f["value"])
                if val == 0:
                    continue  # WT arm of the contrast; direction comes from the alt arm
                cnas[f["column"]] = 1 if val > 0 else -1

        for ct in cancer_types:
            study = STUDY_BY_CANCER_TYPE.get(ct)
            if study is None:
                continue
            for g in muts:
                seen.add(Biomarker(ct, g, "mut", 0, study))
            for g, direction in cnas.items():
                seen.add(Biomarker(ct, g, "cna", direction, study))
    return sorted(seen, key=lambda b: (b.cancer_type, b.gene, b.alt_kind))
