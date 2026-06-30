.PHONY: dev
dev:
	uv sync --all-extras
	uv run pre-commit install

.PHONY: install
install:
	uv sync

.PHONY: test
test:
	uv run pytest src/ tests/

.PHONY: clean
clean:
	find src/ -name "__pycache__" | xargs rm -r
	rm -rf ./build/ .venv/ uv.lock

.PHONY: sandbox-build
sandbox-build:
	docker build -t geryon-sandbox src/geryon/sandbox

.PHONY: sandbox-smoke
sandbox-smoke: sandbox-build
	docker run --rm --network none geryon-sandbox \
		python -c "import duckdb, lifelines, geryon_runtime; print('sandbox OK')"

.PHONY: mypy
mypy:
	uv run --extra dev mypy --check-untyped-defs src/geryon

.PHONY: format
format:
	uv run --extra dev ruff format src/ tests/

.PHONY: etl
etl:
	./scripts/etl.sh

.PHONY: plot
plot:
	./scripts/plot.sh

.PHONY: data
data:
	uv run --extra viewer python scripts/view_data.py

.PHONY: viewer
viewer:
	uv run --extra viewer python -m geryon.cli.viewer

.PHONY: viewer-legacy
viewer-legacy:
	uv run python -m geryon.legacy.cli.annotate

.PHONY: report
report:
	uv run python -m geryon.legacy.cli.report

.PHONY: label-best
label-best:
	uv run python -m geryon.legacy.cli.label_best --aws-profile saml

# Override per invocation, e.g. `make run ITERS=2 PROPOSALS=1` for a quick pass.
ITERS ?= 10
PROPOSALS ?= 3

.PHONY: run
run:
	./scripts/run.sh \
		--provider aws_bedrock \
		--model us.anthropic.claude-opus-4-6-v1 \
		--aws-profile saml \
		--aws-region us-east-2 \
		--max-iterations $(ITERS) \
		--num-proposals $(PROPOSALS) \
		--critic-cycles 1 2>&1 | tee out
		#--model us.anthropic.claude-opus-4-5-20251101-v1:0

# Code-first workflow (LLM writes Python run in the Docker sandbox).
# Requires the sandbox image: run `make sandbox-build` once first.
# Quick first run: `make code-run ITERS=1 PROPOSALS=1`
.PHONY: code-run
code-run:
	uv run python -u -m geryon.codeflow.runner \
		--provider aws_bedrock \
		--model us.anthropic.claude-opus-4-6-v1 \
		--aws-profile saml \
		--aws-region us-east-2 \
		--max-iterations $(ITERS) \
		--num-proposals $(PROPOSALS) \
		--critic-cycles 1 2>&1 | tee code-out

GERYON_DATA_DIR := geryon_data
GERYON_DATA_REPO := git@github.com:kpich/geryon-data.git

.PHONY: schema-context
schema-context:  ## Pre-generate schema_context.txt from confounder_config.json
	uv run python -m geryon.llm.schema_context --write

.PHONY: backup
backup:
	cd $(GERYON_DATA_DIR) && git add -A && \
	git diff --cached --quiet || git commit -m "backup $$(date +%Y-%m-%d_%H:%M:%S)" && \
	git push

.PHONY: restore
restore:
	git clone $(GERYON_DATA_REPO) $(GERYON_DATA_DIR)
