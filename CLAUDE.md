# CYCL Project

Human-in-the-loop LLM tool for hypothesis generation on cancer genomics data. The formal language (`lang/`) captures LLM-generated hypotheses for execution.

## Structure

```
msk_cycl/
├── etl/        # Nextflow pipeline: TSV → parquet
├── lang/       # Formal hypothesis language (what LLM generates)
├── db/         # DuckDB wrapper for parquet files
└── engine/     # Execution: runs hypotheses, returns statistics
```

## Key Pattern: Registry

Add new outcomes/methods by adding to registry - no executor changes needed:

```python
# engine/registry.py
OUTCOME_HANDLERS = {OverallSurvival: OverallSurvivalHandler}
METHOD_IMPLEMENTATIONS = {ComparisonMethod.HAZARD_RATIO_COX: CoxHazardRatioMethod}
```

## Current Status

- One outcome: `OverallSurvival` (OS_MONTHS, OS_STATUS from clinical_patient)
- One method: `HAZARD_RATIO_COX` (simple Cox with cohort indicator only)
- Hypothesis = comparison between two cohorts on an outcome using a statistical method
- LLM generation layer: not implemented yet
