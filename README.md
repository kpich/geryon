Cancer hYpothesis Creation Loop

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

### Reviewing hypotheses

```bash
# List available sessions
python -m msk_cycl.cli.list_sessions --storage-dir cycl_run_outputs/

# Review a session interactively
python -m msk_cycl.cli.review \
  --storage-dir cycl_run_outputs/ \
  --session <session_id> \
  --reviewer "Your Name"

# Export labeled hypotheses to SQLite
python -m msk_cycl.cli.export_db \
  --storage-dir cycl_run_outputs/ \
  --output hypotheses.db
```

### Viewing data

```bash
# Interactive data viewer
uv run harlequin data/msk_solid_heme/
```
