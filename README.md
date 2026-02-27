# HAIIP — Human-Aligned Industrial Intelligence Platform

**Demo video**: https://1drv.ms/v/c/5978203504f409d5/IQALgoMRTYgqT6l8gsL6-nnQAXSjb32DH33UhkqUrdpd3fA?e=sneUNO

[![CI](https://github.com/nextindustriai/haiip/actions/workflows/ci.yml/badge.svg)](https://github.com/nextindustriai/haiip/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/nextindustriai/haiip/branch/main/graph/badge.svg)](https://codecov.io/gh/nextindustriai/haiip)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Limited%20Risk-green)](docs/MODEL_CARD.md)
[![License](https://img.shields.io/badge/license-Proprietary-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)

> **RDI deliverable** — NextIndustriAI project, Centria University of Applied Sciences
> (Jakobstad / Sundsvall / Narvik)

---

## What is HAIIP?

HAIIP is an AI-powered predictive maintenance platform designed for Nordic Small and Medium
Enterprises (SMEs). It combines industrial machine learning, retrieval-augmented generation
(RAG), and EU AI Act compliance in a production-ready multi-tenant system.

**The core problem it solves**: SMEs can't afford unplanned downtime, but also can't afford
to hire data scientists. HAIIP gives them an out-of-the-box AI system that monitors machines,
predicts failures before they happen, explains its reasoning in plain language, and always
keeps a human in the loop.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture](#architecture)
3. [ML System](#ml-system)
4. [Dashboard](#dashboard)
5. [EU AI Act Compliance](#eu-ai-act-compliance)
6. [Quick Start](#quick-start)
7. [Docker](#docker)
8. [API Reference](#api-reference)
9. [Test Suite](#test-suite)
10. [Datasets & Model Performance](#datasets--model-performance)
11. [Research & Experimental Branch](#research--experimental-branch)
12. [RDI Artifacts](#rdi-artifacts)
13. [Citation](#citation)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Anomaly Detection** | IsolationForest on 5-axis sensor readings — AUC-ROC ≥ 0.91 |
| **Predictive Maintenance** | GradientBoosting — 6 failure modes + Remaining Useful Life (RUL) |
| **Drift Detection** | KS test + PSI + Page-Hinkley — detects distribution shift in real time |
| **RAG Document Q&A** | FAISS + sentence-transformers + Groq LLM — ask questions about maintenance manuals |
| **Agentic AI** | ReAct tool-calling agent — natural language industrial diagnosis |
| **ROS2 Closed Loop** | Vibration → AI → Economic decision → machine command → human override (no ROS2 install needed) |
| **Multi-tenancy** | Every dataset, prediction, and audit log is tenant-isolated |
| **JWT Auth** | Access (30 min) + refresh (7 days), RBAC (admin / engineer / viewer) |
| **EU AI Act** | Article 52 compliance — audit log, transparency report, human oversight flag |
| **Streamlit Dashboard** | 10-page industrial HMI, dark theme, demo mode, real-time charts |
| **Background Workers** | Celery + Redis — async retraining, drift checks, cleanup |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HAIIP Platform                              │
│                                                                     │
│  Physical Layer        Application Layer        Presentation Layer  │
│  ─────────────         ─────────────────        ──────────────────  │
│  OPC-UA sensors   ──►  FastAPI (async)      ──►  Streamlit (10 pg)  │
│  MQTT brokers          SQLAlchemy 2.0            Dark industrial UI  │
│  Vibration data        SQLite / PostgreSQL        Demo mode          │
│                                                                     │
│  AI Core                                                            │
│  ────────                                                           │
│  AnomalyDetector     ← IsolationForest + StandardScaler            │
│  MaintenancePredictor ← GradientBoosting (classification + RUL)    │
│  DriftDetector       ← KS / PSI / Page-Hinkley                     │
│  RAGEngine           ← FAISS + sentence-transformers + Groq LLM    │
│  IndustrialAgent     ← ReAct (search, detect, RUL, compliance)     │
│  ComplianceEngine    ← EU AI Act Article 52                        │
│  FeedbackEngine      ← Human-in-the-loop retraining trigger        │
│                                                                     │
│  Workers                                                            │
│  ───────                                                            │
│  Celery + Redis  ← model retraining, drift checks, data cleanup    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ML System

### Anomaly Detection
- **Model**: IsolationForest with StandardScaler preprocessing
- **Input**: 5-axis sensor readings (temperature, rotational speed, torque, tool wear) + optional extra features
- **Output**: `{label, confidence, anomaly_score, explanation}`
- **Performance**: AUC-ROC ≥ 0.91, F1 (macro) ≥ 0.82 on AI4I 2020

### Predictive Maintenance
- **Model**: GradientBoosting (6-class classifier + RUL regressor)
- **Output**: `{label, confidence, failure_probability, rul_cycles}`
- **Performance**: Accuracy ≥ 0.91, MAE ≤ 18 cycles on NASA CMAPSS FD001

### Drift Detection
Three-layer drift monitoring:
```
Layer 1 — KS test          : feature-level statistical drift
Layer 2 — PSI              : population stability index (>0.2 = alert)
Layer 3 — Page-Hinkley     : online sequential changepoint detection
```

### Agentic RAG (ReAct)
```
Query: "Is machine M-003 about to fail?"
  → Intent classification → tool plan
  → search_knowledge_base()    (RAG over maintenance manuals)
  → run_anomaly_detection()    (real-time sensor scoring)
  → calculate_rul()            (remaining useful life estimate)
  → assess_compliance()        (EU AI Act Article 52 check)
  → Synthesised answer + tool trace + confidence score
  → requires_human_review flag (Article 14 — high uncertainty)
```

---

## Dashboard

10-page Streamlit dashboard (`http://localhost:8501`):

| Page | Purpose |
|------|---------|
| Overview | Fleet health, KPI cards, alert summary |
| Machine Detail | Per-machine time series, anomaly history |
| Predictions | Live prediction feed with explanation |
| Alerts | Alert management — acknowledge, filter by severity |
| RAG Q&A | Ask questions about your maintenance documentation |
| AI Agent | Interactive ReAct agent — natural language diagnosis |
| Feedback | Submit corrections to improve the model |
| Drift Monitor | Distribution shift visualization per feature |
| Compliance | EU AI Act audit log + transparency report |
| Admin | User management, model registry, tenant stats |

**Demo mode**: Click "Try Demo" on the login page — no credentials needed, no API required.

**Demo credentials** (when running locally):
```
Tenant:   demo-sme
Email:    admin@haiip.ai
Password: Demo1234!
```

---

## ROS2 Integration — Closed-Loop Industrial Automation

HAIIP ships a full **Human-Aligned Robotic Automation** pipeline out of the box.
It runs in two modes:

| Mode | Requirement | Use case |
|------|-------------|---------|
| **Standalone** (asyncio) | Python only, no ROS2 install | Development, demos, CI |
| **ROS2 nodes** | ros-humble or ros-jazzy | Real robot hardware, production |

### Closed-Loop Architecture

```
[VibrationPublisher]   sensor_msgs/Imu  @  50 Hz
         |   /haiip/vibration/{machine_id}
         v
[InferenceNode]        calls /api/v1/predict (every Nth sample)
         |   /haiip/ai/{machine_id}   label · confidence · anomaly_score
         v
[EconomicNode]         EconomicDecisionEngine  (in-process, < 1 ms)
         |   /haiip/decision/{machine_id}   REPAIR_NOW · SCHEDULE · MONITOR · IGNORE
         v
[ActionNode]           maps decision → machine command
         |   /haiip/command/{machine_id}   STOP · SLOW_DOWN · MONITOR · NOMINAL
         ^
[HumanOverride]        operator console — EU AI Act Art. 14
                       override auto-expires (TTL), AI loop resumes
```

### Run — no ROS2 needed

```bash
# Offline demo (synthetic vibration, no API call):
python -m haiip.ros2.pipeline --no-api

# Live — with running HAIIP API:
python -m haiip.ros2.pipeline

# Fault injection (bearing defect signature):
python -m haiip.ros2.pipeline --fault --machine pump-01

# Human override during demo (interactive console):
#   s = STOP   d = SLOW_DOWN   m = MONITOR   r = RELEASE (return to AI)
```

### Run — with ROS2

```bash
ros2 launch haiip haiip_closed_loop.launch.py machine_id:=pump-01
ros2 launch haiip haiip_closed_loop.launch.py fault_mode:=true

# Inject override from another terminal:
python -m haiip.ros2.human_override --machine pump-01 --command STOP
```

**Why this matters**: Most AI platforms stop at "publish a prediction."
HAIIP closes the loop — the AI decision reaches the machine actuator (STOP / SLOW_DOWN),
the human can override at any time, and the AI resumes automatically when the override expires.
This is the definition of a Human-Aligned Industrial Decision System.

---

## EU AI Act Compliance

HAIIP is classified as **Limited Risk** under EU AI Act Article 52.

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Risk classification (Art. 9) | `ComplianceEngine.classify_risk()` | ✅ |
| Transparency (Art. 52) | Monthly `TransparencyReport` — auto-generated | ✅ |
| Human oversight (Art. 14) | `requires_human_review` flag on every prediction | ✅ |
| Record keeping (Art. 12) | `AuditLog` — every prediction logged, 5-year retention | ✅ |
| Information provision (Art. 13) | Model Card + Dataset Card + in-app disclosure | ✅ |
| Data minimisation (GDPR Art. 5c) | Input features SHA-256 hashed in audit log | ✅ |
| Incident reporting (Art. 73) | `detect_incidents()` — low confidence + bias flags | ✅ |
| Complaint procedure | In-app feedback + DPA contact documented | ✅ |

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com) (free) — for LLM-powered RAG and Agent pages

### Installation

```bash
git clone https://github.com/nextindustriai/haiip.git
cd haiip
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env.local
# Open .env.local and set:
#   GROQ_API_KEY=gsk_...   ← get free key at console.groq.com
#   SECRET_KEY=...          ← any random 32+ character string
```

### Run

```bash
# Terminal 1 — FastAPI backend
PYTHONPATH=. uvicorn haiip.api.main:app --port 8000 --reload

# Terminal 2 — Streamlit dashboard
PYTHONPATH=. streamlit run haiip/dashboard/app.py --server.port 8501
```

Open **http://localhost:8501** — click "Try Demo" or use demo credentials above.

API documentation: **http://localhost:8000/api/docs**

---

## Docker

```bash
# Development (SQLite, hot reload)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production (PostgreSQL, Redis, Prometheus, Grafana)
docker compose up

# Services after startup:
#   Dashboard    → http://localhost:8501
#   API          → http://localhost:8000
#   API Docs     → http://localhost:8000/api/docs
#   Prometheus   → http://localhost:9090
#   Grafana      → http://localhost:3000
```

---

## API Reference

### Auth
```
POST /api/v1/auth/login       → access_token + refresh_token
POST /api/v1/auth/refresh     → new access_token
POST /api/v1/auth/register    → create user (admin only)
GET  /api/v1/auth/me          → current user profile
```

### Predictions
```
POST /api/v1/predict          → single reading → anomaly result
POST /api/v1/predict/batch    → up to 100 readings
GET  /api/v1/predictions      → paginated history (filter by machine)
GET  /api/v1/predictions/{id} → single prediction detail
```

### Alerts
```
GET   /api/v1/alerts                      → list (filter by severity)
POST  /api/v1/alerts                      → create alert
PATCH /api/v1/alerts/{id}/acknowledge     → acknowledge
```

### Feedback
```
POST /api/v1/feedback    → submit correction (triggers retraining check)
GET  /api/v1/feedback    → list submitted feedback
```

### Metrics
```
GET /api/v1/metrics/health           → system health KPIs
GET /api/v1/metrics/machines         → per-machine statistics
GET /api/v1/metrics/alerts/summary   → alert distribution
```

### RAG / Documents
```
POST   /api/v1/query              → question → RAG answer + sources
POST   /api/v1/documents/ingest   → add document to knowledge base
GET    /api/v1/documents/stats    → index statistics
DELETE /api/v1/documents          → clear index
```

### Agent (ReAct)
```
POST /api/v1/agent/query      → natural language → tool-calling diagnosis
GET  /api/v1/agent/capabilities → available tools in this deployment
POST /api/v1/agent/diagnose   → machine_id + readings → full diagnosis
```

### Admin
```
GET    /api/v1/admin/tenant           → tenant info + usage stats
GET    /api/v1/admin/users            → list users
POST   /api/v1/admin/users            → create user
PATCH  /api/v1/admin/users/{id}       → update user role / status
DELETE /api/v1/admin/users/{id}       → deactivate user
GET    /api/v1/admin/models           → model registry
POST   /api/v1/admin/models/{id}/activate → promote model version
GET    /api/v1/admin/stats            → system statistics
GET    /api/v1/audit                  → EU AI Act audit log (exportable)
```

### Example: Single Prediction

```bash
# 1. Login
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"tenant_slug":"demo-sme","email":"admin@haiip.ai","password":"Demo1234!"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "pump-01",
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0
  }'
```

---

## Test Suite

```bash
# Full suite
pytest haiip/tests/ -v --cov=haiip --cov-report=term-missing

# By category
pytest haiip/tests/core/        -v   # unit tests — ML components
pytest haiip/tests/api/         -v   # route tests
pytest haiip/tests/integration/ -v   # end-to-end pipelines
pytest haiip/tests/security/    -v   # OWASP Top 10
pytest haiip/tests/crash/       -v   # NaN / Inf / concurrent edge cases
pytest haiip/tests/features/    -v   # BDD user journeys

# Load tests (requires running API)
locust -f haiip/tests/load/locustfile.py \
  --host=http://localhost:8000 \
  --headless --users 50 --spawn-rate 5 --run-time 60s
```

| Category | Count | Purpose |
|----------|-------|---------|
| Core unit | ~200 | ML models, RAG, compliance, drift, feedback |
| API unit | ~70 | Routes, auth, RBAC, pagination |
| Integration | ~35 | Sensor → prediction → alert pipelines |
| Security (OWASP) | ~30 | SQL injection, JWT, XSS, RBAC bypass |
| Crash/robustness | ~50 | NaN/Inf inputs, concurrency, empty data |
| Feature/BDD | ~40 | Role-based user stories |
| ML evaluation | ~30 | F1, calibration, fairness, reproducibility |
| RAG hallucination | ~30 | Grounding, source citation, uncertainty |
| Load (Locust) | 4 user types | SLA: P95 < 500 ms under 50 concurrent users |

---

## Datasets & Model Performance

### Datasets

| Dataset | Source | License | Used for |
|---------|--------|---------|---------|
| AI4I 2020 | UCI ML Repository | CC BY 4.0 | Failure mode classification, anomaly detection |
| NASA CMAPSS | NASA PrognosticsCOE | Public Domain | Remaining Useful Life prediction |
| CWRU Bearing | Case Western Reserve | Public Domain | Bearing fault detection |
| MIMII | Zenodo / Hitachi | CC BY 4.0 | Sound-based anomaly (planned v0.2) |

See [docs/DATASET_CARD.md](docs/DATASET_CARD.md) for full Gebru et al. (2021) datasheets.

### Model Performance

| Model | Dataset | Metric | Result | Min. Threshold |
|-------|---------|--------|--------|---------------|
| IsolationForest | AI4I 2020 | AUC-ROC | ≥ 0.91 | ≥ 0.85 |
| IsolationForest | AI4I 2020 | F1 (macro) | ≥ 0.82 | ≥ 0.75 |
| GBT Classifier | AI4I 2020 | Accuracy | ≥ 0.91 | ≥ 0.85 |
| GBT Regressor | CMAPSS FD001 | MAE (cycles) | ≤ 18 | ≤ 25 |
| GBT Regressor | CMAPSS FD001 | R² | ≥ 0.82 | ≥ 0.70 |

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for full Mitchell et al. (2019) model card
including intended use, limitations, and bias analysis.

---

## Research & Experimental Branch

The `experimental` branch extends the core platform with Phase 6 research features.
The ROS2 pipeline (above) is included in **both** branches — what experimental adds
on top is the economic decision engine, federated learning, and quantified oversight metrics.

### Core (main) vs Experimental

| Capability | `main` (this branch) | `experimental` |
|------------|----------------------|----------------|
| Anomaly Detection | ✅ IsolationForest | ✅ Same |
| Predictive Maintenance + RUL | ✅ GradientBoosting | ✅ Same |
| RAG Document Q&A | ✅ FAISS + Groq | ✅ Same |
| Agentic ReAct AI | ✅ Tool-calling agent | ✅ Same |
| ROS2 Closed Loop | ✅ Full pipeline | ✅ Same + Economic node |
| Economic Decision Engine | ❌ | ✅ Expected Loss Minimization |
| Federated Learning | ❌ | ✅ FedAvg (3 Nordic SME nodes) |
| Human Oversight Metrics | Qualitative | ✅ HIR / HOG / TCS quantified |
| OpenTelemetry Tracing | ❌ | ✅ Distributed tracing + cost model |
| Kubernetes / Helm | Docker only | ✅ AWS EKS + HPA |
| Terraform IaC | ❌ | ✅ EKS + RDS + ElastiCache |
| Research Notebooks | ❌ | ✅ 2 reproducibility notebooks |

```bash
git checkout experimental

# Closed loop with Economic AI in the loop:
python -m haiip.ros2.pipeline --fault

# Economic decision engine standalone:
python -c "
from haiip.core.economic_ai import EconomicDecisionEngine
d = EconomicDecisionEngine().decide(anomaly_score=0.85, failure_probability=0.80)
print(d.action, f'net benefit: EUR {d.net_benefit:.0f}')
"

# Federated learning:
python -c "
from haiip.core.federated import FederatedLearner
r = FederatedLearner().run(n_rounds=10, local_epochs=3)
print(f'Federated F1: {r.final_global_f1:.4f}  gap: {r.federated_gap:+.4f}')
"
```

**Research questions addressed** (RQ5–RQ8):
- How much cost does Expected Loss Minimization avoid vs naive thresholding?
- Can FedAvg reach F1 within 15% of centralized baseline while preserving privacy?
- What is the measurable Human Override Gain (HOG) at deployment scale?
- What is the compute cost per prediction vs avoided downtime ROI?

---

## RDI Artifacts

| Artifact | Location |
|----------|----------|
| Model Card (Mitchell et al., 2019) | `docs/MODEL_CARD.md` |
| Dataset Card (Gebru et al., 2021) | `docs/DATASET_CARD.md` |
| EU AI Act compliance engine | `haiip/core/compliance.py` |
| Transparency report generator | `ComplianceEngine.generate_transparency_report()` |
| Security tests (OWASP Top 10) | `haiip/tests/security/test_security.py` |
| RAG hallucination test suite | `haiip/tests/core/test_rag_hallucination.py` |
| ML evaluation benchmarks | `haiip/tests/features/test_ml_evaluation.py` |
| Locust load tests | `haiip/tests/load/locustfile.py` |
| CI/CD pipeline | `.github/workflows/ci.yml` |

---

## Citation

```bibtex
@techreport{haiip2025,
  title       = {HAIIP: Human-Aligned Industrial Intelligence Platform},
  author      = {NextIndustriAI Team},
  year        = {2025},
  institution = {Centria University of Applied Sciences},
  note        = {RDI deliverable — NextIndustriAI project
                 (Jakobstad / Sundsvall / Narvik).
                 Experimental Phase 6 features: Economic AI,
                 Federated Learning, Human Oversight quantification,
                 ROS2 closed-loop robotic automation.}
}
```

---

<sub>HAIIP is a research prototype. It is not a certified medical or safety-critical device.
Production deployment requires site-specific calibration and qualified engineering review.</sub>
