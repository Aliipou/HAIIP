# HAIIP — Human-Aligned Industrial Intelligence Platform

RDI-grade AI platform for SME predictive maintenance, anomaly detection, and human-robot collaboration. Built for the NextIndustriAI project across Jakobstad, Sundsvall, and Narvik.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Copy env template
cp .env.example .env.local
# Edit .env.local — set SECRET_KEY, OPENAI_API_KEY, etc.

# 3. Run API (dev)
uvicorn haiip.api.main:app --reload

# 4. API docs → http://localhost:8000/api/docs
```

## Docker (full stack)

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Run tests

```bash
pytest --cov=haiip --cov-report=term-missing
```

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Foundation: FastAPI, JWT, DB, Core AI, Docker |
| 2 | 🔄 Next | RAG engine, OPC UA/MQTT |
| 3 | Planned | Celery workers, MLOps |
| 4 | Planned | Streamlit dashboard |
| 5 | Planned | EU AI Act compliance |

## Architecture

```
haiip/
├── api/       FastAPI — routes, auth, schemas, models
├── core/      AI logic — anomaly, maintenance, drift, feedback, RAG
├── data/      Loaders (AI4I, CWRU, CMAPSS), simulator, OPC UA/MQTT
├── dashboard/ Streamlit UI
├── workers/   Celery background tasks
└── tests/     Mirrors core/ and api/
```
