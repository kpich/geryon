.PHONY: install_precommit_hooks
install_precommit_hooks:
	pip install pre-commit
	pre-commit install

.PHONY: dev
dev:
	pip install -e '.[dev,test,viewer]'

.PHONY: install
install:
	pip install -e .

.PHONY: test
test:
	pytest .

.PHONY: clean
clean:
	find src/ -name "__pycache__" | xargs rm -r
	rm -r ./build/

.PHONY: mypy
mypy:
	mypy --check-untyped-defs src/msk_cycl

.PHONY: etl
etl:
	./scripts/etl.sh

.PHONY: etl-clean
etl-clean:
	rm -rf nextflow/.nextflow* nextflow/work nextflow/pipeline_*

.PHONY: workflow
workflow:
	./scripts/run.sh
