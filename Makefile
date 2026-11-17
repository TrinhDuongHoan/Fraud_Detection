PYTHON := .venv/bin/python
CONFIG ?= configs/default.yaml

.PHONY: install test lint notebooks smoke run prepare train predict
install:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest -q

lint:
	.venv/bin/ruff check src tests scripts notebooks
	.venv/bin/ruff format --check src tests scripts notebooks

notebooks:
	$(PYTHON) scripts/run_notebooks.py

smoke:
	$(PYTHON) -m fraud_detection run --config configs/smoke.yaml

run prepare train predict:
	$(PYTHON) -m fraud_detection $@ --config $(CONFIG)
