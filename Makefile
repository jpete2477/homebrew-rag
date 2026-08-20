# homebrew-rag — everything you need to run, test and deploy this project.
# Run `make` (or `make help`) for the list.

SHELL       := /bin/bash
PYTHON      ?= python3
VENV        ?= venv
BIN         := $(VENV)/bin
RAG         := $(BIN)/homebrew-rag

# Overridable on the command line, e.g. `make ingest DIR=./docs/services TAG=services`
DIR         ?= documents/sample
TAG         ?= demo
Q           ?= What does the audit cost?
TOP_K       ?=
PORT        ?= 8000
HOST        ?= 0.0.0.0
GOLDEN      ?= eval/golden_set.json

TOPK_FLAG   := $(if $(TOP_K),--top-k $(TOP_K),)

.DEFAULT_GOAL := help

# ---------------------------------------------------------------- help ----

.PHONY: help
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "; printf "\nhomebrew-rag targets\n\n"} \
	  /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2} \
	  /^##@/ {printf "\n\033[90m%s\033[0m\n", substr($$0, 5)}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables: DIR=$(DIR) TAG=$(TAG) PORT=$(PORT) GOLDEN=$(GOLDEN)"
	@echo '  Example:   make query Q="what are the payment terms?" TOP_K=8'
	@echo ""

##@ Setup

.PHONY: setup
setup: $(VENV) .env  ## Create the venv, install everything, write .env
	@echo "Ready. Next: make up && make ingest"

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	@echo "Installing CPU-only PyTorch (the default wheel drags in ~2 GB of CUDA)…"
	$(BIN)/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
	$(BIN)/pip install -e ".[local-embeddings,dev]"

.env:
	@cp -n .env.example .env && echo "Wrote .env — add your ANTHROPIC_API_KEY."

.PHONY: install
install:  ## Reinstall dependencies into an existing venv
	$(BIN)/pip install -e ".[local-embeddings,dev]"

##@ Services

.PHONY: up
up:  ## Start Qdrant
	docker compose up -d qdrant
	@printf "Waiting for Qdrant"; \
	for _ in $$(seq 1 30); do \
	  curl -sf http://localhost:6333/readyz >/dev/null && { echo " ready."; exit 0; }; \
	  printf "."; sleep 1; \
	done; echo " timed out." >&2; exit 1

.PHONY: down
down:  ## Stop Qdrant (index volume is preserved)
	docker compose down

.PHONY: logs
logs:  ## Tail container logs
	docker compose logs -f

.PHONY: serve
serve:  ## Run the API + web UI (PORT=8000 by default)
	$(RAG) serve --host $(HOST) --port $(PORT)

.PHONY: dev
dev:  ## Run the API with autoreload
	$(RAG) serve --host 127.0.0.1 --port $(PORT) --reload

##@ Pipeline

.PHONY: ingest
ingest:  ## Index DIR under TAG  (make ingest DIR=./docs/services TAG=services)
	$(RAG) ingest $(DIR) $(TAG)

.PHONY: reingest
reingest:  ## Rebuild TAG from scratch, dropping everything indexed under it
	$(RAG) ingest $(DIR) $(TAG) --replace

.PHONY: query
query:  ## Ask a question  (make query Q="..." TOP_K=8)
	$(RAG) query "$(Q)" $(TOPK_FLAG)

.PHONY: retrieve
retrieve:  ## Show retrieved chunks without calling Claude — costs nothing
	$(RAG) query "$(Q)" $(TOPK_FLAG) --retrieve-only

.PHONY: stats
stats:  ## Show what is currently indexed
	$(RAG) stats

##@ Quality

.PHONY: test
test:  ## Unit tests — no Qdrant, no API key, no PyTorch
	$(BIN)/pytest -v

.PHONY: test-integration
test-integration: up  ## Integration tests against a real Qdrant
	$(BIN)/pytest -m integration -v

.PHONY: test-all
test-all: test test-integration  ## Both suites

.PHONY: eval
eval: $(GOLDEN)  ## Score retrieval against the golden set
	$(RAG) eval $(GOLDEN)

$(GOLDEN):
	@cp -n eval/golden_set.example.json $(GOLDEN) \
	  && echo "Seeded $(GOLDEN) from the example — replace it with your own questions."

.PHONY: lint
lint:  ## Lint and check formatting
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

.PHONY: format
format:  ## Autoformat and autofix
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

.PHONY: check
check: lint test  ## What CI runs

##@ Deploy

.PHONY: docker-build
docker-build:  ## Build the API image
	docker compose --profile api build

.PHONY: docker-up
docker-up:  ## Run Qdrant + the API in containers
	docker compose --profile api up -d --build
	@echo "API on http://localhost:8000 — check: curl -s localhost:8000/health"

.PHONY: docker-down
docker-down:  ## Stop both containers
	docker compose --profile api down

.PHONY: install-service
install-service:  ## Install the systemd unit (edit User/paths first)
	sudo cp deploy/rag-api.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable --now rag-api
	systemctl status rag-api --no-pager

##@ Housekeeping

.PHONY: clean
clean:  ## Remove build artefacts and caches (keeps the venv and the index)
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov src/*.egg-info
	find . -path ./venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf

.PHONY: clean-index
clean-index:  ## DESTRUCTIVE: delete the Qdrant volume and everything indexed
	docker compose down -v

.PHONY: clean-all
clean-all: clean  ## Also remove the venv
	rm -rf $(VENV)
