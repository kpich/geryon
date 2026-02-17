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

.PHONY: int-test
int-test:
	uv run pytest tests/

.PHONY: clean
clean:
	find src/ -name "__pycache__" | xargs rm -r
	rm -rf ./build/ .venv/ uv.lock

.PHONY: mypy
mypy:
	uv run --extra dev mypy --check-untyped-defs src/msk_cycl

.PHONY: lint
lint:
	uv run --extra dev ruff check src/

.PHONY: format
format:
	uv run --extra dev ruff format src/

.PHONY: etl
etl:
	./scripts/etl.sh

.PHONY: profile
profile:
	uv run python -m msk_cycl.etl.profiler --dir $$(ls -d ~/data/msk_cycle_data/*/ | tail -1)

.PHONY: etl-clean
etl-clean:
	rm -rf nextflow/.nextflow* nextflow/work nextflow/pipeline_*

.PHONY: data
data:
	uv run --extra viewer python scripts/view_data.py

.PHONY: annotate
annotate:
	uv run python -m msk_cycl.cli.annotate

.PHONY: workflow
workflow:
	./scripts/run.sh
