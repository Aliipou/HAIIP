
# HAIIP — Human-Aligned Industrial Intelligence Platform
Demo video : https://1drv.ms/v/c/5978203504f409d5/IQALgoMRTYgqT6l8gsL6-nnQAXSjb32DH33UhkqUrdpd3fA?e=sneUNO
[![CI](https://github.com/nextindustriai/haiip/actions/workflows/ci.yml/badge.svg)](https://github.com/nextindustriai/haiip/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/nextindustriai/haiip/branch/main/graph/badge.svg)](https://codecov.io/gh/nextindustriai/haiip)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Limited%20Risk%20%E2%9C%85-green)](docs/MODEL_CARD.md)
[![License](https://img.shields.io/badge/license-Proprietary-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)
=======
# HAIIP Experimental Branch — Phase 6 Research Features

> **Status**: Research-grade — validated, tested, not production-hardened.
> **Branch**: `main` (Phase 6 features are included in main after Phase 4/5 foundation)
>>>>>>> 608b5de6889868aad6ac9248273448f7af13815b

---

## Overview

Phase 6 extends HAIIP with four research-grade features that push the platform
from a production-ready SME tool toward an RDI-grade academic contribution
suitable for peer-reviewed publication.

---

## Feature Comparison: Main vs Phase 6

<<<<<<< HEAD
1. [Research Motivation](#research-motivation)
2. [Architecture](#architecture)
3. [ML System Overview](#ml-system-overview)
4. [Phase 6 — Economic AI, Federated Learning & Human Oversight](#phase-6--economic-ai-federated-learning--human-oversight)
5. [EU AI Act Compliance](#eu-ai-act-compliance)
6. [Quick Start](#quick-start)
7. [Full Stack (Docker)](#full-stack-docker)
8. [Cloud Deployment (Kubernetes)](#cloud-deployment-kubernetes)
9. [Test Suite](#test-suite)
10. [API Reference](#api-reference)
11. [Datasets](#datasets)
12. [Model Performance](#model-performance)
13. [RDI Artifacts](#rdi-artifacts)
14. [Contributing](#contributing)
15. [Citation](#citation)
=======
| Capability | Main (Phase 1–5) | Phase 6 (Experimental) |
|------------|------------------|------------------------|
| Anomaly Detection | ✅ Production | ✅ Same |
| Predictive Maintenance | ✅ Production | ✅ Same |
| RAG Q&A | ✅ Production | ✅ Same |
| Agentic RAG | ✅ ReAct agent | ✅ Same |
| Economic Decision | ❌ | ✅ Expected Loss Minimization |
| Federated Learning | ❌ | ✅ FedAvg (3 Nordic nodes, simulated) |
| Human Oversight Metrics | Qualitative (Art. 14) | ✅ HIR / HOG / TCS quantified |
| OpenTelemetry Tracing | ❌ | ✅ Distributed tracing + SLA checks |
| Per-Prediction Cost Model | ❌ | ✅ Compute cost vs avoided downtime |
| Kubernetes | Docker only | ✅ EKS deployment |
| Helm Chart | ❌ | ✅ Production-ready chart |
| Terraform IaC | ❌ | ✅ AWS EKS + RDS + ElastiCache |
| Research Notebooks | ❌ | ✅ 2 reproducibility notebooks |
>>>>>>> 608b5de6889868aad6ac9248273448f7af13815b

---

## Research Questions Addressed

### RQ5 — Economic AI
> How much downtime cost does the Expected Loss Minimization engine avoid
> compared to naive probability thresholding?

**Implementation**: `haiip/core/economic_ai.py`
**Test coverage**: `haiip/tests/core/test_economic_ai.py` (35+ tests)
**Notebook**: `notebooks/01_economic_decision.ipynb`

<<<<<<< HEAD
**Key research questions (main branch)**:
1. Can IsolationForest + GradientBoosting achieve >85% F1 on real SME sensor data?
2. What is the minimum human oversight rate to satisfy EU AI Act Article 14?
3. How does RAG-based document querying improve maintenance engineer decision-making?
4. What is the performance/compliance trade-off in privacy-preserving anomaly logging?
=======
**Formula**:
```
E[Cost_wait]   = P(failure) × C_downtime × safety_factor
E[Cost_action] = P(no_failure) × C_false_positive + C_maintenance
Net_benefit    = E[Cost_wait] − E[Cost_action]
```

**Decision rules**:
- `REPAIR_NOW` : P(failure) ≥ 0.75
- `SCHEDULE`   : P(failure) ≥ 0.50
- `MONITOR`    : anomaly_score ≥ 0.20
- `IGNORE`     : below noise floor
>>>>>>> 608b5de6889868aad6ac9248273448f7af13815b

**Extended research questions (Phase 6 — experimental)**:
5. How much downtime cost does the Expected Loss Minimization engine avoid vs naive thresholding?
6. Can FedAvg across 3 Nordic SME nodes achieve F1 within 15% of the centralized baseline while preserving privacy?
7. What is the measurable Human Override Gain (HOG) and Trust Calibration Score (TCS) for this system?
8. What is the compute cost per prediction vs avoided downtime at fleet scale?

> **Experimental branch notice**: Phase 6 features (Federated Learning, Economic AI, Human Oversight quantification)
> are research-grade implementations. They are validated, tested, and documented but not production-hardened.
> See `docs/EXPERIMENTAL_BRANCH.md` for the full research scope comparison.

---

### RQ6 — Federated Learning
> Can FedAvg across 3 Nordic SME nodes achieve F1 within 15% of the centralized
> baseline while preserving privacy?

**Implementation**: `haiip/core/federated.py`
**Test coverage**: `haiip/tests/core/test_federated.py` (20+ tests)
**Notebook**: `notebooks/02_federated_learning.ipynb`

**Nodes**:
| Node | Country | Industry | n_samples | failure_rate |
|------|---------|----------|-----------|--------------|
| SME_FI | Finland | Paper mill | 800 | 12% |
| SME_SE | Sweden | Automotive stamping | 1200 | 8% |
| SME_NO | Norway | Offshore pumps | 600 | 18% |

<<<<<<< HEAD
| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI + asyncio | REST API, JWT auth, multi-tenancy |
| Database | SQLAlchemy 2.0 async + SQLite/PostgreSQL | Persistence |
| Anomaly Detection | IsolationForest + StandardScaler | Unsupervised anomaly scoring |
| Maintenance Prediction | GradientBoosting (6-class + RUL) | Failure mode + life estimation |
| Drift Detection | KS test + PSI + Page-Hinkley | Distribution shift monitoring |
| RAG Engine | FAISS + sentence-transformers + OpenAI | Maintenance document Q&A |
| Agentic RAG | ReAct pattern (Yao et al., 2022) | Tool-calling industrial AI agent |
| Background Workers | Celery + Redis | Retraining, drift checks, cleanup |
| Dashboard | Streamlit | Industrial HMI with dark theme |
| Compliance | Custom (Article 52) | EU AI Act audit trail |
| **Economic AI** | Expected Loss Minimization | Maintenance decision optimisation (€) |
| **Federated Learning** | FedAvg (McMahan et al., 2017) | Privacy-preserving cross-SME learning |
| **Human Oversight** | HIR / HOG / TCS metrics | Quantified human oversight (Art. 14) |
| **Observability** | OpenTelemetry + cost model | Distributed tracing + ROI per prediction |
| **Infrastructure** | Kubernetes + Helm + Terraform | Cloud-native AWS EKS deployment |
=======
**Privacy guarantees**: Only weight deltas transmitted — no raw data leaves node boundary.
>>>>>>> 608b5de6889868aad6ac9248273448f7af13815b

---

### RQ7 — Human Oversight Quantification
> What is the measurable Human Override Gain (HOG) and Trust Calibration Score (TCS)?

**Implementation**: `haiip/core/human_oversight.py`
**Test coverage**: `haiip/tests/core/test_human_oversight.py` (30+ tests)

**Metrics**:
| Metric | Formula | Target |
|--------|---------|--------|
| HIR (Human Intervention Rate) | `|reviewed| / |decisions|` | 0.05–0.15 |
| HOG (Human Override Gain) | `F1(corrected) − F1(ai_only)` | > 0.02 |
| TCS (Trust Calibration Score) | `1 − ECE` | ≥ 0.80 |
| ECE (Expected Calibration Error) | Guo et al. (2017) | < 0.10 |

### Agentic RAG — ReAct Tool-Calling Agent (Phase 5)
```
Query: "Is machine M-003 about to fail?"
       → Intent classification → tool plan
       → search_knowledge_base()    (RAG over manuals)
       → run_anomaly_detection()    (real-time sensor score)
       → calculate_rul()            (remaining useful life)
       → assess_compliance()        (EU AI Act Article 52)
       → Synthesised answer + tool trace + confidence
       → requires_human_review flag (Article 14)
```

---

## Phase 6 — Economic AI, Federated Learning & Human Oversight

> **Research-grade features** — validated, tested, not production-hardened.
> Full scope in [docs/EXPERIMENTAL_BRANCH.md](docs/EXPERIMENTAL_BRANCH.md).

### Economic Decision Engine
Transforms ML outputs into cost-optimal maintenance decisions using Expected Loss Minimization:

```
E[Cost_wait]   = P(failure) × C_downtime × safety_factor
E[Cost_action] = P(no_failure) × C_false_positive + C_maintenance
Net_benefit    = E[Cost_wait] − E[Cost_action]

Decision → REPAIR_NOW | SCHEDULE | MONITOR | IGNORE
```

Nordic SME defaults: `C_downtime = €4,000/failure`, `C_maintenance = €590/event`

### Federated SME Learning (FedAvg)
Privacy-preserving collaborative learning across 3 Nordic manufacturing sites:

| Node | Country | Industry | Dataset size |
|------|---------|----------|-------------|
| SME_FI | Finland | Paper mill (vibration) | 800 samples |
| SME_SE | Sweden | Automotive stamping | 1,200 samples |
| SME_NO | Norway | Offshore pumps | 600 samples |

**Privacy**: Only weight deltas transmitted — no raw sensor data leaves site boundary.
**Quality**: Federated F1 within 15% of centralized baseline (validated).

### Human Oversight Quantification (EU AI Act Art. 14)

| Metric | Formula | Target |
|--------|---------|--------|
| HIR (Intervention Rate) | `reviewed / decisions` | 0.05–0.15 |
| HOG (Override Gain) | `F1(corrected) − F1(ai_only)` | > 0.02 |
| TCS (Trust Calibration) | `1 − ECE` (Guo et al., 2017) | ≥ 0.80 |

### Observability — Per-Prediction Cost Model
```
compute_cost = inference_ms × (GPU_rate / 3_600_000)
avoided_cost = P(failure) × C_downtime − C_maintenance   [if P > 0.5]
net_value    = avoided_cost − compute_cost

At SME scale (t3.medium, eu-north-1):
  Cost/prediction: ~€0.000007
  ROI ratio:       >10,000× for high-accuracy models
```

```python
from haiip.core.economic_ai import EconomicDecisionEngine
from haiip.core.federated import FederatedLearner
from haiip.core.human_oversight import HumanOversightEngine

# Economic decision
engine = EconomicDecisionEngine()
d = engine.decide(anomaly_score=0.85, failure_probability=0.80)
print(d.action, f"€{d.net_benefit:.0f} net benefit")

# Federated learning
result = FederatedLearner().run(n_rounds=10, local_epochs=3)
print(f"Federated F1: {result.final_global_f1:.4f}  gap: {result.federated_gap:+.4f}")
```

---

### RQ8 — Per-Prediction Cost Model
> What is the compute cost per prediction vs avoided downtime at fleet scale?

**Implementation**: `haiip/observability/cost_model.py`
**Test coverage**: `haiip/tests/core/test_observability.py`

**At SME defaults (AWS eu-north-1)**:
- Inference cost: ~€0.000007 per prediction (t3.medium)
- Avoided downtime per correct REPAIR_NOW: €4,000+
- ROI ratio: >10,000× for high-accuracy models

---

## Limitations and Threats to Validity

1. **Federated simulation**: Federation is simulated in-process (no actual network). Real-world
   communication delays, Byzantine failures, and gradient attacks are not modelled.

2. **Synthetic non-IID data**: Node data is synthesised from Gaussian distributions with
   configurable shift. Real SME sensor data may have different statistical properties.

3. **Economic parameters**: Default cost parameters (€500/hr, 8h MTTR) are representative
   Nordic SME estimates — actual values vary by site and must be calibrated per deployment.

4. **Oversight simulation**: HOG/TCS metrics are computed on simulated human decisions,
   not real operator data. A longitudinal field study is required for validation.

---

## How to Run Phase 6 Features

```bash
# Economic decision
python -c "
from haiip.core.economic_ai import EconomicDecisionEngine
engine = EconomicDecisionEngine()
d = engine.decide(anomaly_score=0.85, failure_probability=0.80)
print(d.action, d.net_benefit)
"

# Federated learning
python -c "
from haiip.core.federated import FederatedLearner
result = FederatedLearner().run(n_rounds=5, local_epochs=2)
print(f'F1: {result.final_global_f1:.4f}, gap: {result.federated_gap:+.4f}')
"

# Human oversight
python -c "
from haiip.core.human_oversight import HumanOversightEngine, OversightEvent
eng = HumanOversightEngine()
eng.record(OversightEvent.create('d1','failure',0.8,'normal',True,True,'normal'))
m = eng.compute_metrics()
print(f'HIR={m.hir:.2f}, HOG={m.hog:+.4f}, TCS={m.tcs:.4f}')
"

# Notebooks
jupyter lab notebooks/
```

<<<<<<< HEAD
**Demo mode** (no API required — works offline):
```bash
streamlit run haiip/dashboard/app.py
# Click "Try Demo" on login page
```

---

## Full Stack (Docker)

```bash
# Development (hot reload, SQLite)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production (PostgreSQL, Redis, Prometheus, Grafana)
docker compose up

# Services:
#   API          → http://localhost:8000
#   Dashboard    → http://localhost:8501
#   API Docs     → http://localhost:8000/api/docs
#   Prometheus   → http://localhost:9090
#   Grafana      → http://localhost:3000
```

---

## Cloud Deployment (Kubernetes)

Phase 6 ships a complete cloud-native deployment stack targeting AWS EKS (eu-north-1 — GDPR compliant).

```bash
# 1. Provision infrastructure (Terraform)
cd terraform/
terraform init
terraform apply -var="environment=production"

# 2. Deploy with Helm
helm dependency update helm/haiip/
helm upgrade --install haiip helm/haiip/ \
  --namespace haiip --create-namespace \
  -f helm/haiip/values.yaml

# 3. Or apply raw Kubernetes manifests
kubectl apply -f k8s/
```

**Infrastructure components:**

| Component | Technology | Notes |
|-----------|-----------|-------|
| Compute | AWS EKS 1.29 | 2–10 API pods, HPA auto-scaling |
| Database | AWS RDS PostgreSQL 15 | Multi-AZ, encrypted, 7-day backups |
| Cache | AWS ElastiCache Redis 7 | Session + Celery broker |
| Storage | AWS S3 + EFS | Model artifacts (versioned), model cache |
| Ingress | nginx-ingress + cert-manager | TLS via Let's Encrypt |
| Observability | OpenTelemetry → OTLP | Grafana Tempo / Jaeger compatible |
| Secrets | External Secrets Operator | AWS Secrets Manager integration |

---

## Test Suite

HAIIP maintains a comprehensive multi-tier test pyramid targeting 100% production-grade coverage.

```bash
# Full test suite
pytest haiip/tests/ -v --cov=haiip --cov-report=term-missing

# Unit tests (fast, no I/O)
pytest haiip/tests/core/ haiip/tests/dashboard/ -v

# API integration tests
pytest haiip/tests/api/ haiip/tests/integration/ -v

# Security tests (OWASP Top 10)
pytest haiip/tests/security/ -v

# Crash / robustness tests (NaN, Inf, concurrent, edge cases)
pytest haiip/tests/crash/ -v

# Feature tests (BDD user journeys)
pytest haiip/tests/features/ -v

# RAG hallucination tests
pytest haiip/tests/core/test_rag_hallucination.py -v

# Load tests (requires running API at localhost:8000)
locust -f haiip/tests/load/locustfile.py \
       --host=http://localhost:8000 \
       --headless --users 50 --spawn-rate 5 --run-time 60s

# Phase 6 load tests
locust -f haiip/tests/load/locustfile_phase6.py \
       --host=http://localhost:8000 \
       --headless --users 50 --spawn-rate 5 --run-time 60s
```

### Test Coverage by Category

| Category | Tests | Purpose |
|----------|-------|---------|
| Core unit | ~200 | ML components, economic AI, federated, oversight, compliance |
| API unit | ~70 | Route validation, auth, RBAC, agent endpoints |
| Integration | ~35 | End-to-end pipelines: sensor→economic→oversight |
| Security (OWASP) | ~30 | SQL injection, JWT, RBAC, XSS |
| Crash/robustness | ~50 | NaN/Inf/empty, concurrent access, extreme values |
| Feature/BDD | ~40 | Role-based user stories, fleet ROI, federated scenarios |
| ML evaluation | ~30 | F1, calibration, fairness, reproducibility |
| RAG hallucination | ~30 | Grounding, source citation, uncertainty |
| Observability | ~20 | Cost model, OTel tracer, SLA breach detection |
| Load (Locust) | 7 user types | SLA validation, economic/agent endpoints |

---

## API Reference

### Authentication
```
POST /api/v1/auth/login          # Login → access_token + refresh_token
POST /api/v1/auth/refresh        # Refresh access token
POST /api/v1/auth/register       # Register new user
GET  /api/v1/auth/me             # Current user info
```

### Predictions
```
POST /api/v1/predict             # Single prediction (anomaly + maintenance)
POST /api/v1/predict/batch       # Batch prediction
GET  /api/v1/predictions         # List predictions (paginated)
GET  /api/v1/predictions/{id}    # Single prediction detail
```

### Alerts
```
GET    /api/v1/alerts            # List alerts (filterable by severity)
POST   /api/v1/alerts            # Create alert
GET    /api/v1/alerts/{id}       # Single alert
PATCH  /api/v1/alerts/{id}/acknowledge  # Acknowledge alert
```

### Feedback (Human-in-the-loop)
```
POST /api/v1/feedback            # Submit prediction feedback
GET  /api/v1/feedback            # List submitted feedback
```

### Metrics
```
GET /api/v1/metrics/health       # System health KPIs
GET /api/v1/metrics/machines     # Per-machine statistics
GET /api/v1/metrics/alerts/summary  # Alert distribution summary
```

### Document Q&A (RAG)
```
POST /api/v1/query               # Ask a question about maintenance docs
POST /api/v1/documents/ingest    # Ingest a document into RAG index
GET  /api/v1/documents/stats     # RAG index statistics
DELETE /api/v1/documents         # Clear RAG index
```

### Agent (Phase 5 — Agentic RAG)
```
POST /api/v1/agent/query         # Natural language industrial AI query (ReAct agent)
GET  /api/v1/agent/capabilities  # Discover available tools in this deployment
POST /api/v1/agent/diagnose      # machine_id + sensor readings → full agentic diagnosis
```

### Economic AI (Phase 6)
```
POST /api/v1/economic/decide     # Single cost-optimal maintenance decision
POST /api/v1/economic/batch      # Batch decisions for fleet
GET  /api/v1/economic/roi        # Fleet ROI summary for reporting period
```

### Admin (admin role required)
```
GET  /api/v1/admin/tenant        # Tenant information + stats
GET  /api/v1/admin/users         # List users
POST /api/v1/admin/users         # Create user
PATCH /api/v1/admin/users/{id}   # Update user
DELETE /api/v1/admin/users/{id}  # Deactivate user
GET  /api/v1/admin/models        # Model registry
POST /api/v1/admin/models/{id}/activate  # Activate model version
GET  /api/v1/admin/stats         # System statistics
GET  /api/v1/audit               # EU AI Act audit log
```

---

## Datasets

| Dataset | Source | License | Used for |
|---------|--------|---------|---------|
| AI4I 2020 | UCI ML Repository | CC BY 4.0 | Failure mode classification |
| NASA CMAPSS | NASA PrognosticsCOE | Public Domain | RUL prediction |
| CWRU Bearing | Case Western Reserve | Public Domain | Bearing fault detection |
| MIMII | Zenodo (Hitachi) | CC BY 4.0 | Sound anomaly (planned v0.2) |

See [docs/DATASET_CARD.md](docs/DATASET_CARD.md) for full datasheets.

---

## Model Performance

| Model | Dataset | Metric | Value | Threshold |
|-------|---------|--------|-------|-----------|
| IsolationForest | AI4I 2020 | AUC-ROC | ≥0.91 | ≥0.85 |
| IsolationForest | AI4I 2020 | F1 (macro) | ≥0.82 | ≥0.75 |
| GBT Classifier | AI4I 2020 | Accuracy | ≥0.91 | ≥0.85 |
| GBT Regressor | CMAPSS FD001 | MAE (cycles) | ≤18 | ≤25 |
| GBT Regressor | CMAPSS FD001 | R² | ≥0.82 | ≥0.70 |
| FedAvg (Phase 6) | Synthetic (3 nodes) | F1 vs centralized | gap < 0.15 | gap < 0.20 |
| Economic Engine (Phase 6) | Simulated fleet | Decision accuracy | ≥0.88 | ≥0.80 |
| Trust Calibration (Phase 6) | Simulated oversight | TCS (1−ECE) | ≥0.80 | ≥0.70 |

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for full model card including bias analysis.
See [docs/EXPERIMENTAL_BRANCH.md](docs/EXPERIMENTAL_BRANCH.md) for Phase 6 research metrics.

---

## RDI Artifacts

This project is a Centria RDI deliverable. The following artifacts are available:

| Artifact | Location | Description |
|----------|----------|-------------|
| Model Card | `docs/MODEL_CARD.md` | Mitchell et al. (2019) format |
| Dataset Card | `docs/DATASET_CARD.md` | Gebru et al. (2021) format |
| Experimental Branch | `docs/EXPERIMENTAL_BRANCH.md` | Phase 6 RQ5-8 scope + limitations |
| Compliance Engine | `haiip/core/compliance.py` | EU AI Act Article 52 |
| Transparency Report | `ComplianceEngine.generate_transparency_report()` | Auto-generated monthly |
| Audit Trail | `haiip/api/routes/admin.py` | Immutable, exportable CSV |
| ML Evaluation | `haiip/tests/features/test_ml_evaluation.py` | Reproducible benchmarks |
| Hallucination Tests | `haiip/tests/core/test_rag_hallucination.py` | RAG grounding verification |
| Security Tests | `haiip/tests/security/test_security.py` | OWASP Top 10 |
| Load Tests | `haiip/tests/load/locustfile.py` | SLA validation (Phase 1–5) |
| Phase 6 Load Tests | `haiip/tests/load/locustfile_phase6.py` | Economic AI + Agent SLA |
| Economic AI Notebook | `notebooks/01_economic_decision.ipynb` | Decision boundary + fleet ROI |
| Federated Learning Notebook | `notebooks/02_federated_learning.ipynb` | FedAvg learning curve + gap analysis |
| K8s Manifests | `k8s/` | Production Kubernetes deployment |
| Helm Chart | `helm/haiip/` | Parameterised deployment chart |
| Terraform IaC | `terraform/` | AWS EKS + RDS + ElastiCache |

=======
>>>>>>> 608b5de6889868aad6ac9248273448f7af13815b
---

## Citation

If using Phase 6 experimental features in research, please cite:

```bibtex
@techreport{haiip2025,
  title  = {HAIIP: Human-Aligned Industrial Intelligence Platform},
  author = {NextIndustriAI Team},
  year   = {2025},
  institution = {Centria University of Applied Sciences},
  note   = {RDI deliverable — NextIndustriAI project (Jakobstad/Sundsvall/Narvik)}
}
```
