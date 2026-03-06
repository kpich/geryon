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

.PHONY: mypy
mypy:
	uv run --extra dev mypy --check-untyped-defs src/geryon

.PHONY: format
format:
	uv run --extra dev ruff format src/

.PHONY: etl
etl:
	./scripts/etl.sh

.PHONY: etl-clean
etl-clean:
	rm -rf nextflow/.nextflow* nextflow/work nextflow/pipeline_*

.PHONY: plot
plot:
	./scripts/plot.sh

.PHONY: plot-clean
plot-clean:
	rm -rf plots/

.PHONY: data
data:
	uv run --extra viewer python scripts/view_data.py

.PHONY: viewer
viewer:
	uv run python -m geryon.cli.annotate

.PHONY: report
report:
	uv run python -m geryon.cli.report

.PHONY: label-best
label-best:
	uv run python -m geryon.cli.label_best --aws-profile saml

.PHONY: run
run:
	./scripts/run.sh \
		--provider aws_bedrock \
		--model us.anthropic.claude-opus-4-5-20251101-v1:0 \
		--aws-profile saml \
		--aws-region us-east-2 \
		--max-iterations 10 \
		--num-proposals 3 \
		--critic-cycles 1 2>&1 | tee out

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
