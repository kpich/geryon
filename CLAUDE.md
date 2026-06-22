# Geryon Project

Human-in-the-loop LLM tool for hypothesis generation on cancer genomics data. LLM proposes hypotheses in formal language, executor runs them, LLM narrates results.

## Structure

```
geryon/
├── etl/        # Nextflow pipeline: TSV → parquet (includes CNA transpose)
├── lang/       # Formal hypothesis language (GeryonHyp spec)
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

## Testing

- **Unit tests** go in `src/` next to the code they test (e.g., `executor_test.py`). Use mocks for external dependencies. Run with `make test`.
- **Integration tests** go in `tests/integration/`. These use real parquet files, real DB connections, etc. Run with `make int-test`.

## Code Style

**No vacuous comments** - Don't add comments that just restate what the code does. Only comment if adding non-obvious context (why, not what)

## Safety — Never Invoke LLM Workflows

NEVER run commands that trigger external LLM API calls (Bedrock, OpenAI, Anthropic, Ollama, etc.). This includes:
- `run.sh`, `scripts/run.sh`, or any wrapper that launches a session
- `uv run python -m geryon` or any direct invocation of the workflow
- Any script or command that calls the LLM providers in `geryon/llm/`

These calls are billable and should only be triggered by the human operator. Stick to code edits, tests, and read-only exploration.

## Safety — Never Perform Git Operations

NEVER run git commands that change repository state, and never ask to. The human operator handles all git themselves. This includes (non-exhaustive):
- `git commit`, `git add`, `git push`, `git pull`, `git merge`, `git rebase`, `git reset`, `git stash`, `git cherry-pick`
- Creating, deleting, renaming, switching, or checking out branches (`git branch`, `git checkout`, `git switch`)
- Creating tags, editing git config, or anything else that mutates the repo or its history

Do not do any of this of your own volition, and do not ask whether you should — just leave git alone entirely. Read-only inspection (`git status`, `git log`, `git diff`, `git show`) is fine. If a task seems to need a commit, branch, or other git action, stop and let the human do it.

## Safety — Never Delete Files Without Asking

NEVER delete or remove files without first explicitly asking the human and getting a clear yes. This includes `rm`, `rm -rf`, `mv` that overwrites or discards a file, `git clean`, truncating/emptying a file, or any command whose effect is to remove or destroy a file's contents.

A user saying they don't need a file, don't want to keep it, or that it's temporary is NOT permission to delete it — that is context, not an instruction to act. Ask first ("want me to delete X?") and wait for an explicit yes before removing anything. When in doubt, leave the file in place and let the human delete it themselves.
