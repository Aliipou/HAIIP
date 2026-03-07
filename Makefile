# HAIIP — Makefile
# Targets for development, testing, and operations.
# Run `make help` to see all available commands.

.DEFAULT_GOAL := help
SHELL         := bash
PYTHON        := python
PYTEST        := pytest
UVICORN_PORT  := 8000
STREAMLIT_PORT:= 8501
PYTHONPATH    := .

# ── Colours ───────────────────────────────────────────────────────────────────
RESET  := \033[0m
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BOLD)Usage:$(RESET)\n  make $(CYAN)<target>$(RESET)\n\n$(BOLD)Targets:$(RESET)\n"} \
	     /^[a-zA-Z_-]+:.*##/ { printf "  $(CYAN)%-26s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
.PHONY: install
install: ## Install all dependencies (dev + extras)
	pip install -e ".[dev]"

.PHONY: install-prod
install-prod: ## Install production dependencies only
	pip install -e "."

.PHONY: env
env: ## Copy .env.example → .env.local (edit before use)
	@[ -f .env.local ] && echo ".env.local already exists — skipping" || cp .env.example .env.local
	@echo "$(YELLOW)Edit .env.local and set your secrets before starting the server.$(RESET)"

# ── Development servers ───────────────────────────────────────────────────────
.PHONY: dev
dev: ## Start FastAPI dev server (hot-reload, port 8000)
	PYTHONPATH=$(PYTHONPATH) uvicorn haiip.api.main:app \
	    --host 0.0.0.0 --port $(UVICORN_PORT) --reload \
	    --log-level info

.PHONY: dashboard
dashboard: ## Start Streamlit dashboard (port 8501)
	PYTHONPATH=$(PYTHONPATH) python -m streamlit run haiip/dashboard/app.py \
	    --server.port $(STREAMLIT_PORT) --server.address 0.0.0.0

.PHONY: worker
worker: ## Start Celery worker (requires Redis)
	PYTHONPATH=$(PYTHONPATH) celery -A haiip.workers.tasks worker \
	    --loglevel=info --concurrency=4 -Q default,ml_ops,retraining

.PHONY: beat
beat: ## Start Celery Beat scheduler (auto-retrain every 6h)
	PYTHONPATH=$(PYTHONPATH) celery -A haiip.workers.tasks beat \
	    --loglevel=info --schedule=/tmp/celerybeat-schedule

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test: ## Run full test suite (parallel, verbose)
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/ \
	    -n auto --dist=worksteal \
	    -v --tb=short \
	    --asyncio-mode=auto

.PHONY: test-fast
test-fast: ## Run tests excluding slow integration/load tests
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/ \
	    -n auto --dist=worksteal \
	    -v --tb=short \
	    --asyncio-mode=auto \
	    --ignore=haiip/tests/load \
	    --ignore=haiip/tests/integration \
	    -m "not slow"

.PHONY: test-core
test-core: ## Run core unit tests only
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/core/ \
	    -n auto -v --tb=short --asyncio-mode=auto

.PHONY: test-api
test-api: ## Run API route tests only
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/api/ \
	    -n auto -v --tb=short --asyncio-mode=auto

.PHONY: test-workers
test-workers: ## Run Celery worker tests only
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/workers/ \
	    -n auto -v --tb=short --asyncio-mode=auto

.PHONY: test-integration
test-integration: ## Run integration tests (full pipeline)
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/integration/ \
	    -v --tb=long --asyncio-mode=auto

.PHONY: test-security
test-security: ## Run OWASP security tests
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/security/ \
	    -v --tb=short --asyncio-mode=auto

.PHONY: test-crash
test-crash: ## Run crash/edge-case robustness tests
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/crash/ \
	    -v --tb=short --asyncio-mode=auto

.PHONY: coverage
coverage: ## Run tests with coverage report (HTML + terminal)
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) haiip/tests/ \
	    -n auto --dist=worksteal \
	    --asyncio-mode=auto \
	    --cov=haiip \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    --cov-fail-under=80
	@echo "$(GREEN)Coverage report: htmlcov/index.html$(RESET)"

# ── Code quality ──────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Run ruff linter
	ruff check haiip/

.PHONY: format
format: ## Auto-format with ruff
	ruff format haiip/

.PHONY: typecheck
typecheck: ## Run mypy type checks
	mypy haiip/ --ignore-missing-imports --no-error-summary

.PHONY: security-scan
security-scan: ## Run bandit security scan
	bandit -r haiip/ -ll -x haiip/tests/

.PHONY: check
check: lint typecheck security-scan ## Run all static checks (lint + types + security)

# ── ML Operations ─────────────────────────────────────────────────────────────
.PHONY: retrain
retrain: ## Trigger manual retraining for default tenant (requires running API + Redis)
	@TENANT=$${TENANT:-default}; \
	 TOKEN=$${TOKEN:-}; \
	 echo "$(CYAN)Triggering retraining for tenant: $$TENANT$(RESET)"; \
	 curl -s -X POST http://localhost:$(UVICORN_PORT)/api/v1/ml-ops/retrain \
	     -H "Content-Type: application/json" \
	     -H "Authorization: Bearer $$TOKEN" \
	     -d "{\"tenant_id\": \"$$TENANT\", \"force_reason\": \"manual\"}" | python -m json.tool

.PHONY: export-onnx
export-onnx: ## Export champion anomaly model to ONNX (requires running API + Redis)
	@TENANT=$${TENANT:-default}; \
	 TOKEN=$${TOKEN:-}; \
	 echo "$(CYAN)Exporting ONNX model for tenant: $$TENANT$(RESET)"; \
	 curl -s -X POST http://localhost:$(UVICORN_PORT)/api/v1/ml-ops/export-onnx \
	     -H "Content-Type: application/json" \
	     -H "Authorization: Bearer $$TOKEN" \
	     -d "{\"tenant_id\": \"$$TENANT\", \"model_type\": \"anomaly\", \"opset\": 17}" | python -m json.tool

.PHONY: benchmark
benchmark: ## Run ONNX latency benchmark (p50/p95/p99, SLA=50ms)
	@TENANT=$${TENANT:-default}; \
	 TOKEN=$${TOKEN:-}; \
	 echo "$(CYAN)Running benchmark for tenant: $$TENANT$(RESET)"; \
	 curl -s -X POST http://localhost:$(UVICORN_PORT)/api/v1/ml-ops/benchmark \
	     -H "Content-Type: application/json" \
	     -H "Authorization: Bearer $$TOKEN" \
	     -d "{\"tenant_id\": \"$$TENANT\", \"model_type\": \"anomaly\", \"n_runs\": 200}" | python -m json.tool

.PHONY: pipeline-status
pipeline-status: ## Get ML pipeline status (artifacts, drift, SLA)
	@TOKEN=$${TOKEN:-}; \
	 curl -s http://localhost:$(UVICORN_PORT)/api/v1/ml-ops/pipeline-status \
	     -H "Authorization: Bearer $$TOKEN" | python -m json.tool

.PHONY: model-versions
model-versions: ## List active model versions
	@TOKEN=$${TOKEN:-}; \
	 curl -s http://localhost:$(UVICORN_PORT)/api/v1/ml-ops/model-versions \
	     -H "Authorization: Bearer $$TOKEN" | python -m json.tool

# ── Docker ────────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build: ## Build all Docker images
	docker compose build

.PHONY: docker-up
docker-up: ## Start full stack (API + worker + dashboard + Redis + Prometheus + Grafana)
	docker compose up -d
	@echo "$(GREEN)Stack started:$(RESET)"
	@echo "  API:        http://localhost:8000/api/docs"
	@echo "  Dashboard:  http://localhost:8501"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000  (admin / admin)"

.PHONY: docker-down
docker-down: ## Stop all containers
	docker compose down

.PHONY: docker-down-v
docker-down-v: ## Stop all containers and remove volumes
	docker compose down -v

.PHONY: docker-logs
docker-logs: ## Tail logs from all containers
	docker compose logs -f --tail=100

.PHONY: docker-logs-api
docker-logs-api: ## Tail API container logs only
	docker compose logs -f api --tail=200

.PHONY: docker-ps
docker-ps: ## Show running container status
	docker compose ps

.PHONY: docker-restart-api
docker-restart-api: ## Restart only the API container
	docker compose restart api

# ── Health & Observability ────────────────────────────────────────────────────
.PHONY: health
health: ## Check API health endpoint
	@curl -s http://localhost:$(UVICORN_PORT)/health | python -m json.tool

.PHONY: metrics
metrics: ## Show raw Prometheus metrics (first 60 lines)
	@curl -s http://localhost:$(UVICORN_PORT)/metrics | grep "^haiip_" | head -60

.PHONY: api-info
api-info: ## Show API version, features, and SLA targets
	@curl -s http://localhost:$(UVICORN_PORT)/api/v1/ | python -m json.tool

# ── Database ──────────────────────────────────────────────────────────────────
.PHONY: db-shell
db-shell: ## Open SQLite shell on dev database
	sqlite3 haiip_dev.db

.PHONY: db-reset
db-reset: ## Delete dev database (next startup recreates tables)
	@read -p "Delete haiip_dev.db? This is irreversible. [y/N] " confirm; \
	 [ "$$confirm" = "y" ] && rm -f haiip_dev.db && echo "Deleted." || echo "Cancelled."

# ── Load testing ──────────────────────────────────────────────────────────────
.PHONY: load-test
load-test: ## Run Locust load test (headless, 20 users, 60s)
	PYTHONPATH=$(PYTHONPATH) locust \
	    -f haiip/tests/load/locustfile.py \
	    --host=http://localhost:$(UVICORN_PORT) \
	    --headless -u 20 -r 5 -t 60s \
	    --csv=load_test_results \
	    --only-summary

.PHONY: load-test-ui
load-test-ui: ## Start Locust web UI (http://localhost:8089)
	PYTHONPATH=$(PYTHONPATH) locust \
	    -f haiip/tests/load/locustfile.py \
	    --host=http://localhost:$(UVICORN_PORT) \
	    --web-port=8089

# ── Git / Release ─────────────────────────────────────────────────────────────
.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

.PHONY: clean
clean: ## Remove build artefacts, caches, coverage data
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	@echo "$(GREEN)Clean complete.$(RESET)"

.PHONY: clean-models
clean-models: ## Remove all trained model artefacts (use with care)
	@read -p "Delete model_artifacts/? [y/N] " confirm; \
	 [ "$$confirm" = "y" ] && rm -rf model_artifacts/ && echo "Deleted." || echo "Cancelled."
