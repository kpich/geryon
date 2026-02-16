# CYCL Project

Human-in-the-loop LLM tool for hypothesis generation on cancer genomics data. LLM proposes hypotheses in formal language, executor runs them, LLM narrates results.

## Structure

```
msk_cycl/
├── etl/        # Nextflow pipeline: TSV → parquet (includes CNA transpose)
├── lang/       # Formal hypothesis language (CyclHyp spec)
├── db/         # DuckDB wrapper for parquet files
├── engine/     # Execution engine (registry-based for outcomes/methods)
├── llm/        # LLM integration (generator, narrator, providers, schema)
├── labeling/   # JSONL storage for labeled hypotheses
└── workflow/   # Linear workflow: propose → execute → narrate → store
```

## Key Patterns

**Registry for extensibility** - Add new outcomes/methods without executor changes:
```python
# engine/registry.py
OUTCOME_HANDLERS = {OverallSurvival: OverallSurvivalHandler}
METHOD_IMPLEMENTATIONS = {ComparisonMethod.HAZARD_RATIO_COX: CoxHazardRatioMethod}
```

**Structured LLM output** - Uses Instructor library for Pydantic validation with retry

**Resilient workflow** - Empty proposals/failures don't crash session, logs to file

## Current Status

- One outcome: `OverallSurvival` (Cox regression on OS_MONTHS/OS_STATUS)
- One method: `HAZARD_RATIO_COX` (cohort A vs B)
- LLM providers: AWS Bedrock (default), OpenAI, Anthropic
- Hypothesis storage: JSONL files (one per session)
- Very early stage - schema, prompts, error handling all subject to change

## Code Style

**No vacuous comments** - Don't add comments that just restate what the code does. Only comment if adding non-obvious context (why, not what)

## Safety — Never Invoke LLM Workflows

NEVER run commands that trigger external LLM API calls (Bedrock, OpenAI, Anthropic, Ollama, etc.). This includes:
- `run.sh`, `scripts/run.sh`, or any wrapper that launches a session
- `uv run python -m msk_cycl` or any direct invocation of the workflow
- Any script or command that calls the LLM providers in `msk_cycl/llm/`

These calls are billable and should only be triggered by the human operator. Stick to code edits, tests, and read-only exploration.
