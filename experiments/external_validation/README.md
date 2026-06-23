# External validation (preliminary)

Isolated experiment — **imports** `geryon` (the Cox method) but changes nothing in
`src/geryon`. Asks whether the genomic biomarkers underlying Geryon's MSK
hypotheses carry an independent overall-survival signal in TCGA.

## Why a *relaxed* question

The current corpus (66 hypotheses) is **entirely `survival_from_treatment`** —
treatment-anchored on MSK timelines (IO/chemo response). TCGA has no comparable
treatment/response data, so **no hypothesis ports as-written**. What *does* port is
the genomic stratifier inside each one (gene × alteration × cancer type).

So we test the weaker, independent claim:

> Does *altered vs wild-type* for that biomarker associate with OS in the matching
> TCGA PanCancer Atlas cohort?

This is **not** "the treatment effect replicates." It drops treatment context and
asks only about marginal prognostic association. It is a genuinely independent
cohort (different patients, assay, population), so it is still stronger evidence
than the within-MSK `val q<0.05` filter — just for a different, narrower claim.

## How it reuses Geryon

- HRs come from `geryon.engine.methods.CoxHazardRatioMethod` — the same statistical
  core the live engine uses.
- `OS_STATUS` is parsed with Geryon's `"1:DECEASED"` convention.
- **Standard (non-truncated) Cox** is used on purpose: TCGA time-zero is diagnosis,
  so the sequencing-anchored left-truncation Geryon applies to MSK-IMPACT does not
  apply here.

## Data

cBioPortal public REST API (`www.cbioportal.org/api`, no auth), `*_tcga_pan_can_atlas_2018`
studies. Fetches only the genes the corpus references — no bulk downloads. Mapping
MSK `CANCER_TYPE` → TCGA study lives in `biomarkers.py` (`STUDY_BY_CANCER_TYPE`).

## Run

```bash
cd experiments/external_validation
PYTHONPATH=. ../../.venv/bin/python run.py          # -> out/results.csv
PYTHONPATH=. ../../.venv/bin/python plot_forest.py  # -> out/forest.png
../../.venv/bin/python -m pytest biomarkers_test.py # offline logic tests
```

## Files

| File | Role |
|------|------|
| `cbioportal.py` | stdlib REST client (genes, OS, mutations, CNA) |
| `biomarkers.py` | extract gene×alteration×cancer from corpus + cancer-type→study map |
| `run.py` | fetch + Cox per biomarker → `out/results.csv` (with BH q-values) |
| `plot_forest.py` | forest plot → `out/forest.png` |
| `biomarkers_test.py` | offline unit tests for extraction / selection logic |

## Caveats for the poster

- **Relaxed claim** (prognostic association, not treatment-effect replication).
- One study per cancer type; NSCLC→LUAD only (LUSC excluded); Esophagogastric→STAD.
- Guards: ≥5 altered patients and ≥3 events/arm, else skipped (e.g. FOXA1/PRAD has
  too few OS events).
- Co-mutation / derived-cohort logic is reduced to single-gene marginal contrasts.
- BH-FDR is over this frozen biomarker set, so q-values are honest confirmatory.
