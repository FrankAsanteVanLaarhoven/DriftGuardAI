# DriftGuard — developer entrypoints. Everything routes through `uv`.
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Keep DriftGuard's venv hermetic: don't let a system PYTHONPATH (e.g. a sourced
# ROS setup) leak foreign site-packages and pytest plugins into our environment.
unexport PYTHONPATH

IMAGE ?= driftguard:local
PORT ?= 8000
SERVICE_URL ?= http://localhost:$(PORT)

.PHONY: help install lock lint fmt test data train run drift docker stack stack-down demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install pinned deps (core + dev)
	uv sync --extra dev

lock: ## Regenerate the lockfile
	uv lock

lint: ## Ruff lint
	uv run ruff check .

fmt: ## Ruff format
	uv run ruff format .

test: ## Run unit + integration + fallback tests
	uv run pytest

data: ## Build the fixed, seeded processed dataset from HF ag_news
	uv run python -m driftguard.data

train: ## Train primary + baseline, register in MLflow, write baseline gate metrics
	uv run python -m driftguard.train

run: ## Serve the FastAPI app locally with the fallback contract
	uv run uvicorn driftguard.api.main:app --host 0.0.0.0 --port $(PORT)

drift: ## Run the PSI drift check against a sample (non-zero exit on drift)
	uv run python -m driftguard.drift artifacts/current_shifted.json

gate: ## Baseline gate: fail (exit 1) if the candidate regresses vs baseline
	uv run python -m driftguard.gate

drift-text: ## Text-aware composite drift (PSI + domain-classifier) on a sample
	uv run python -m driftguard.textdrift artifacts/current_shifted.json

benchmark: ## Run the controlled drift-injection benchmark (detection rate + FPR)
	uv run python benchmarks/eval_harness.py

benchmark-sweep: ## Severity sweep (detection boundary) for gradual_topic drift
	uv run python benchmarks/eval_harness.py --sweep gradual_topic

docker: ## Build the production image
	docker build -t $(IMAGE) .

stack: ## Launch app + Prometheus + Grafana + MLflow via docker compose
	docker compose up -d --build

stack-down: ## Tear down the local stack
	docker compose down -v

demo: ## End-to-end local proof (see README "Demo script")
	@bash scripts/demo.sh

clean: ## Remove local caches and generated artifacts (keeps committed fallback)
	rm -rf .pytest_cache .ruff_cache mlruns mlflow.db data/*.parquet artifacts/primary.joblib
