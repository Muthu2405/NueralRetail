# NeuralRetail — top-level Makefile
#
# Targets run inside the .venv created by `make install`. The `python`
# used for everything is the venv's own interpreter, so package version
# resolution stays stable (no surprise upgrades between targets).
#
# Override on the command line, e.g. `make test PY=python3`.

PY     ?= .venv/Scripts/python.exe
UV     ?= python -m uv
PYTHON ?= 3.12
PORT_API ?= 8000
PORT_DASH ?= 8501

.PHONY: help install venv data features train api dashboard test pipeline clean nuke lint api-test dashboard-test monitor

help:
	@echo "NeuralRetail — make targets"
	@echo "  install          Create .venv with uv and install all deps (incl. dev)"
	@echo "  venv             Just create the .venv (no installs)"
	@echo "  data             Ingest raw data and write data/processed/cleaned.parquet"
	@echo "  features         Build feature parquet files (RFM, daily aggregates)"
	@echo "  train            Train all models and log to MLflow"
	@echo "  monitor          Phase 7: generate Evidently data-drift HTML report"
	@echo "  api              Run the FastAPI scoring service on :$(PORT_API)"
	@echo "  dashboard        Run the Streamlit dashboard on :$(PORT_DASH)"
	@echo "  test             Run pytest (all suites)"
	@echo "  api-test         Run only the API smoke tests"
	@echo "  dashboard-test   Run only the dashboard smoke tests (AppTest + hex-code guard)"
	@echo "  pipeline         data -> features -> train -> monitor"
	@echo "  lint             Run ruff over src/ and tests/"
	@echo "  clean            Remove build / cache artefacts"
	@echo "  nuke             Remove .venv, mlruns, models, processed (destructive!)"

venv:
	$(UV) venv --python $(PYTHON) .venv

install: venv
	$(UV) pip install --python .venv -e ".[dev]"
	@if not exist .env ( copy .env.example .env >NUL )

data:
	$(PY) -m neuralretail.cli data

features:
	$(PY) -m neuralretail.cli features

train:
	$(PY) -m neuralretail.cli train

monitor:
	$(PY) -m neuralretail.cli monitor

api:
	$(PY) -m uvicorn neuralretail.api.main:app --host 0.0.0.0 --port $(PORT_API) --reload

dashboard:
	$(PY) -m streamlit run src/neuralretail/dashboard/app.py --server.port $(PORT_DASH)

test:
	$(PY) -m pytest

api-test:
	$(PY) -m pytest tests/test_api.py -v

dashboard-test:
	$(PY) -m pytest tests/test_dashboard.py -v

pipeline: data features train monitor

lint:
	$(PY) -m ruff check src tests

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

nuke: clean
	rm -rf .venv mlruns models data/processed/*.parquet data/processed/*.json
