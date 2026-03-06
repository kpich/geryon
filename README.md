# Geryon

v much in dev/beta -- check back later hopefully.

LLM tool for exploring cancer clinicogenomics data and generating hypotheses.
Explore data,
possibly generate derived views,
propose hypotheses in a small formal language,
execute,
get effect size + pval,
interpret,
iterate.


## Setup

```bash
make dev   # uv sync --all-extras + pre-commit install
```

Data (parquet files) goes in `~/data/geryon_data/`. The workflow auto-detects
the latest subdirectory.

## Use

```bash
make run                                                # aws_bedrock, default model + settings
./scripts/run.sh --provider anthropic --model claude-sonnet-4-6
./scripts/run.sh --provider openai    --model gpt-4o
./scripts/run.sh --max-iterations 5 --num-proposals 3 --critic-cycles 1
```

Providers: `aws_bedrock` (default), `anthropic`, `openai`.
AWS Bedrock setup: [notebook](https://github.com/clinical-data-mining/llm_examples/blob/main/notebooks/04.Setting_Up_ClaudeCode_with_AWS_Bedrock.ipynb).

Sessions are written to `geryon_data/sessions/`.

```bash
make data      # manually examine raw ETL/derived data (harlequin viewer)
make viewer    # browse + human-annotate hypotheses (http://localhost:8765)
make report    # generate static HTML report from session
make label-best  # auto-label best hypotheses via LLM
```

Labels are saved to `geryon_data/labeled/`.

## ETL

```bash
make etl        # Nextflow pipeline: TSV → parquet + .profile.json per table
make etl-clean  # remove Nextflow working files
```

## Development

```bash
make test      # unit + integration tests
make mypy
make format
make backup    # push geryon_data to its git remote
make restore   # clone geryon_data from remote
```
