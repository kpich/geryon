Geryon

## Setup

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Install dependencies

```bash
# Install all dependencies (creates .venv automatically)
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

Alternatively, use the Makefile:
```bash
make dev
```

### Running the workflow

```bash
# Run hypothesis generation with local model
./scripts/run.sh --model llama3.1:8b --max-iterations 2

# Or with Claude
export ANTHROPIC_API_KEY="your-key"
./scripts/run.sh --provider anthropic --model claude-3-5-sonnet-20241022
```

### Development

```bash
# Run tests
uv run pytest
# Or: make test

# Run linting
uv run ruff check src/
# Or: make lint

# Run type checking
uv run mypy src/
# Or: make mypy

# Format code
uv run ruff format src/
# Or: make format
```

### Annotating hypotheses

```bash
# Launch browser-based annotator (recommended)
make annotate

# Or with options:
uv run python -m geryon.cli.annotate \
  --output-dir geryon_run_outputs/ \
  --labeled-dir labeled_hypotheses/ \
  --port 8765
```

Opens a local page at `http://localhost:8765` that auto-discovers all session JSONLs under `geryon_run_outputs/`, shows unlabeled hypotheses newest-first with cohort descriptions, stats, and narrative, and lets you label via radio buttons. Labels are saved as individual JSON files in `labeled_hypotheses/`.
### Viewing data

```bash
# Interactive data viewer
uv run harlequin data/msk_solid_heme/
```

### AWS etc

[ipynb about configuring aws bedrock etc](https://github.com/clinical-data-mining/llm_examples/blob/main/notebooks/04.Setting_Up_ClaudeCode_with_AWS_Bedrock.ipynb)
