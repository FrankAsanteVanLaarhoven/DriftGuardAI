# DriftGuard — developer entrypoints. Everything routes through `uv`.
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Keep DriftGuard's venv hermetic: don't let a system PYTHONPATH (e.g. a sourced
# ROS setup) leak foreign site-packages and pytest plugins into our environment.
unexport PYTHONPATH

IMAGE ?= driftguard:local
PORT ?= 8000
SERVICE_URL ?= http://localhost:$(PORT)

# The `docker` first on PATH may be a wrapper that injects a `compose` subcommand — turning
# `docker compose ...` into `compose compose ...`. Prefer the real CLI (override: make stack DOCKER=...).
DOCKER := $(shell [ -x /usr/bin/docker ] && echo /usr/bin/docker || command -v docker)

.PHONY: help install lock lint fmt test data train train-transformer run run-transformer \
	drift benchmark benchmark-sweep benchmark-stream benchmark-h2h recovery recovery-sweep \
	example-tabular example-embedding docker stack stack-down demo clean

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

train-transformer: ## Fine-tune, gate, and promote the DistilBERT primary (GPU; reproduces macro-F1 0.9412)
	uv run --extra transformer python scripts/train_distilbert.py \
		--epochs 3 --batch-size 32 --max-length 128 --promote

run: ## Serve the FastAPI app locally with the fallback contract
	uv run uvicorn driftguard.api.main:app --host 0.0.0.0 --port $(PORT)

run-transformer: ## Serve with the transformer extra (serves the promoted DistilBERT bundle); still falls back to baseline
	uv run --extra transformer uvicorn driftguard.api.main:app --host 0.0.0.0 --port $(PORT)

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

benchmark-stream: ## Streaming drift benchmark: detection latency across temporal patterns
	uv run python benchmarks/streaming.py

benchmark-h2h: ## Head-to-head: DriftGuard vs Evidently vs NannyML (+ scipy KS baseline)
	uv run --extra bench python benchmarks/head_to_head.py

recovery: ## Closed-loop: detect -> retrain -> gate under concept drift (timed, measured)
	uv run python benchmarks/closed_loop.py

recovery-sweep: ## Recovery-vs-severity curve, mean±std over seeds (recovery + retention + TTR)
	uv run python benchmarks/closed_loop.py --sweep-p 0.3,0.5,0.7,0.9 --seeds 3 --train-sample 40000

example-tabular: ## Second reference instance: the governance framework on Adult (tabular)
	uv run python examples/tabular_adult.py

example-embedding: ## Third reference instance: the framework on MiniLM embeddings (20 News)
	uv run --extra embed python examples/embedding_20news.py

docker: ## Build the production image
	$(DOCKER) build -t $(IMAGE) .

stack: ## Launch app + Prometheus + Grafana + MLflow (DRIFTGUARD_APP_PORT=8010 to avoid a busy 8000)
	$(DOCKER) compose up -d --build

stack-down: ## Tear down the local stack
	$(DOCKER) compose down -v

demo: ## End-to-end local proof (see README "Demo script")
	@bash scripts/demo.sh

clean: ## Remove local caches and generated artifacts (keeps committed fallback)
	rm -rf .pytest_cache .ruff_cache mlruns mlflow.db data/*.parquet artifacts/primary.joblib
