.PHONY: dev
dev:
	uv sync --all-extras
	uv run pre-commit install

.PHONY: install
install:
	uv sync

.PHONY: test
test:
	uv run pytest src/

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
	uv run --extra dev ruff format src/

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

# Override per invocation, e.g. `make run ITERS=2` for a quick pass.
ITERS ?= 10

# Code-first workflow (LLM writes Python run in the Docker sandbox).
# Requires the sandbox image: run `make sandbox-build` once first.
# Quick first run: `make run ITERS=1`
.PHONY: run
run:
	uv run python -u -m geryon.codeflow.runner \
		--aws-profile saml \
		--max-iterations $(ITERS) \
		--critic-cycles 1 2>&1 | tee out

GERYON_DATA_DIR := geryon_data
GERYON_DATA_REPO := git@github.com:kpich/geryon-data.git

# Commit message for `make backup`. Defaults to a timestamp; override for a
# meaningful checkpoint, e.g.:
#   make backup BACKUP_MSG="wholesale swap: drop legacy sessions/, code_sessions -> sessions"
BACKUP_MSG ?= backup $(shell date +%Y-%m-%d_%H:%M:%S)

.PHONY: backup
backup:
	cd $(GERYON_DATA_DIR) && git add -A && \
	git diff --cached --quiet || git commit -m "$(BACKUP_MSG)" && \
	git push

.PHONY: restore
restore:
	git clone $(GERYON_DATA_REPO) $(GERYON_DATA_DIR)
