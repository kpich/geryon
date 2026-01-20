.PHONY: dev
dev:
	uv sync --all-extras
	uv run pre-commit install

.PHONY: install
install:
	uv sync

.PHONY: test
test:
	uv run pytest .

.PHONY: clean
clean:
	find src/ -name "__pycache__" | xargs rm -r
	rm -rf ./build/ .venv/ uv.lock

.PHONY: mypy
mypy:
	uv run mypy --check-untyped-defs src/msk_cycl

.PHONY: lint
lint:
	uv run ruff check src/

.PHONY: format
format:
	uv run ruff format src/

.PHONY: etl
etl:
	./scripts/etl.sh

.PHONY: etl-clean
etl-clean:
	rm -rf nextflow/.nextflow* nextflow/work nextflow/pipeline_*

.PHONY: workflow
workflow:
	./scripts/run.sh
