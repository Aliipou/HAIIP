# HAIIP — Deployment Guide

## Stack

| Component | Technology | Port |
|---|---|---|
| API | FastAPI + Uvicorn | 8000 |
| Dashboard | Streamlit | 8501 |
| Worker | Celery + Redis | — |
| Database | SQLite (dev) / PostgreSQL (prod) | 5432 |
| Message broker | Redis | 6379 |
| Metrics | Prometheus | 9090 |
| Dashboards | Grafana | 3000 |
| Reverse proxy | NGINX | 80 / 443 |

---

## Local Development

### Prerequisites

- Python 3.11+
- Redis (or `docker compose up redis`)
- Git

### Setup

```bash
git clone https://github.com/Aliipou/HAIIP.git
cd HAIIP
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env.local       # fill in secrets
```

### Start services

```bash
# Terminal 1 — API
make dev

# Terminal 2 — Celery worker
make worker

# Terminal 3 — Dashboard
make dashboard
```

API docs: http://localhost:8000/api/docs
Dashboard: http://localhost:8501

---

## Docker Compose (recommended for demos and staging)

```bash
# Build images
make docker-build

# Start full stack
make docker-up

# Tail logs
make docker-logs

# Stop
make docker-down
```

Services started:
- `api` — FastAPI on :8000
- `worker` — Celery on Redis
- `beat` — Celery Beat scheduler (auto-retrain every 6h)
- `dashboard` — Streamlit on :8501
- `redis` — message broker + cache
- `prometheus` — metrics scraper on :9090
- `grafana` — dashboards on :3000 (admin/admin on first run)

### Environment variables (`.env.local`)

```env
APP_ENV=development
SECRET_KEY=<random-64-char-hex>
DATABASE_URL=sqlite+aiosqlite:///./haiip_dev.db
REDIS_URL=redis://localhost:6379/0
GROQ_API_KEY=<your-groq-key>
PROMETHEUS_ENABLED=true
MODEL_ARTIFACTS_PATH=model_artifacts
DRIFT_THRESHOLD=0.2
```

---

## Production Deployment

### Minimum server spec

| Resource | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Disk | 50 GB SSD | 200 GB SSD |
| GPU | None (CPU ONNX) | NVIDIA T4 (optional) |

### PostgreSQL setup

```bash
# In .env.production
DATABASE_URL=postgresql+asyncpg://haiip:PASSWORD@db:5432/haiip_prod
```

Run alembic migrations (or let `create_all_tables()` run on first startup in staging):
```bash
PYTHONPATH=. python -c "import asyncio; from haiip.api.database import create_all_tables; asyncio.run(create_all_tables())"
```

### NGINX + TLS

The repo includes `nginx/nginx.conf` with:
- TLS termination (Let's Encrypt / self-signed)
- Rate limiting: 100 req/s per IP
- Security headers: HSTS, X-Frame-Options, CSP
- Upstream to Uvicorn on :8000

### Kubernetes (Helm)

```bash
# Add secrets
kubectl create secret generic haiip-secrets \
  --from-literal=SECRET_KEY=... \
  --from-literal=DATABASE_URL=...

# Deploy
helm upgrade --install haiip helm/haiip/ \
  -f helm/haiip/values.yaml \
  --set image.tag=$(git rev-parse --short HEAD)
```

HPA is configured for API deployment: min 2, max 10 pods, target CPU 70%.

---

## CI/CD Pipeline (GitHub Actions)

### `.github/workflows/ci.yml` — runs on every push and PR

| Job | What it does |
|---|---|
| `lint` | ruff check + ruff format --check |
| `typecheck` | mypy |
| `security` | bandit -ll |
| `test-core` | pytest haiip/tests/core/ -n auto |
| `test-api` | pytest haiip/tests/api/ -n auto |
| `test-integration` | pytest haiip/tests/integration/ |
| `test-security` | pytest haiip/tests/security/ |
| `coverage` | pytest --cov, fail under 80% |
| `docker-build` | docker build --no-cache (verify image builds) |
| `pre-commit` | all hooks on changed files |

### `.github/workflows/cd.yml` — runs on push to `main`

1. Build multi-arch Docker image (`linux/amd64`, `linux/arm64`)
2. Push to GitHub Container Registry (`ghcr.io/aliipou/haiip`)
3. Tag with git SHA and `latest`
4. Trigger ArgoCD sync (if configured)

---

## Branch Strategy

| Branch | Purpose | Protected? |
|---|---|---|
| `main` | Stable, prod-ready code | Yes — requires PR + CI pass |
| `feat/*` | New features | No |
| `fix/*` | Bug fixes | No |
| `chore/*` | Deps, config, refactor | No |

All PRs to `main` require:
- At least 1 reviewer approval
- All CI jobs green
- No secrets detected (detect-secrets pre-commit hook)

---

## Rollback Procedure

```bash
# API: roll back to previous image
docker pull ghcr.io/aliipou/haiip:PREVIOUS_SHA
docker compose up -d api

# Model: roll back champion
# In Python console or via admin endpoint:
from haiip.core.model_registry import _registry
_registry["default"]["anomaly_detector"] = previous_version_entry

# Database: restore from backup
pg_restore -d haiip_prod backup_YYYYMMDD.dump
```

Full rollback runbook: `docs/ML_OPS_RUNBOOK.md`
