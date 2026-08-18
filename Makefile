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

# Extra nextflow args. Name a data version and point at its own source tree, e.g.:
#   make etl ARGS="--version medonc-pfs-2026-08 --data_root ~/data/msk-impact/msk_solid_heme_medonc"
# Without ARGS the version defaults to today's date, as it always has.
ARGS ?=

.PHONY: etl
etl:
	./scripts/etl.sh $(ARGS)

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

# Line of investigation. `main` is the open-ended chain; a named chain is defined by
# chains/<name>.md (focus prose + the data version it runs on) and sees only its own
# prior hypotheses. E.g. `make run CHAIN=medonc-pfs ITERS=5`.
CHAIN ?= main

# Code-first workflow (LLM writes Python run in the Docker sandbox).
# Requires the sandbox image: run `make sandbox-build` once first.
# Quick first run: `make run ITERS=1`
.PHONY: run
run:
	uv run python -u -m geryon.codeflow.runner \
		--aws-profile saml \
		--max-iterations $(ITERS) \
		--chain $(CHAIN) \
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
