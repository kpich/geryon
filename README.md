# Geryon

Early-stage LLM tool for hypothesis generation on cancer clinicogenomics data.
Proposes hypotheses in a formal language, executes them statistically, and narrates
the results.

## Setup

```bash
make dev   # uv sync --all-extras + pre-commit install
```

Data (parquet files) goes in `~/data/geryon_data/`. The workflow auto-detects
the latest subdirectory.

## Running

```bash
make run                                                # aws_bedrock, default model + settings
./scripts/run.sh --provider anthropic --model claude-sonnet-4-6
./scripts/run.sh --provider openai    --model gpt-4o
./scripts/run.sh --max-iterations 5 --num-proposals 3 --critic-cycles 1
```

Providers: `aws_bedrock` (default), `anthropic`, `openai`.
AWS Bedrock setup: [notebook](https://github.com/clinical-data-mining/llm_examples/blob/main/notebooks/04.Setting_Up_ClaudeCode_with_AWS_Bedrock.ipynb).

Sessions are written to `geryon_data/sessions/`.

## Annotating

```bash
make annotate   # opens http://localhost:8765
```

Labels are saved to `geryon_data/labeled/`.

## ETL

```bash
make etl        # Nextflow pipeline: TSV → parquet + .profile.json per table
make etl-clean  # remove Nextflow working files
```

## Development

```bash
make test      # unit tests (src/)
make int-test  # integration tests (tests/)
make lint
make mypy
make format
make data      # interactive parquet viewer (harlequin)
make backup    # push geryon_data to its git remote
make restore   # clone geryon_data from remote
```
