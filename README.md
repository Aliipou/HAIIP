<div align="center">

# HAIIP
### Human-Aligned Industrial Intelligence Platform

**Production-grade industrial AI for Nordic SMEs**

[![CI](https://github.com/Aliipou/HAIIP/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/HAIIP/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-73%25-brightgreen)](https://github.com/Aliipou/HAIIP)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Limited%20Risk%20%E2%80%94%20Art.%2052-green)](docs/MODEL_CARD.md)
[![Python](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Proprietary-blue)](LICENSE)

> **RDI deliverable** — NextIndustriAI project
> Centria University of Applied Sciences · Jakobstad / Sundsvall / Narvik

---

### [Watch  Demo](https://1drv.ms/v/c/5978203504f409d5/IQALgoMRTYgqT6l8gsL6-nnQAXSjb32DH33UhkqUrdpd3fA?e=sneUNO) · [API Docs](http://localhost:8000/api/docs) · [Model Card](docs/MODEL_CARD.md) · [Dataset Card](docs/DATASET_CARD.md)

</div>

---

## What Problem Does HAIIP Solve?

Nordic SMEs lose an average of **€500–€2,000 per hour** of unplanned machine downtime. They cannot afford dedicated data scientists. They cannot deploy black-box AI that operators distrust. And from August 2026, any AI system they use must comply with the EU AI Act.

HAIIP gives an SME a deployable system that:

- Monitors machines in real time via OPC UA, MQTT, or sensor CSV
- Predicts failures before they happen and explains *why* in plain language
- Keeps a human in the loop at every decision — operator can always override
- Closes the loop: AI decision reaches the machine actuator, not just a dashboard
- Generates the research evidence an RDI project needs to report to EU funders

**One command to start. One hour to onboard a new SME.**

---

## Architecture

```
+--------------------------------------------------------------------------+
|                            HAIIP Platform                                |
|                                                                          |
|   Data Sources            AI Core                  Interfaces            |
|   -----------             -------                  ----------            |
|   OPC UA (PLC)    -->     AnomalyDetector          Streamlit (10 pages)  |
|   MQTT broker     -->     MaintenancePredictor      Dark industrial HMI   |
|   Vibration CSV   -->     DriftDetector             Demo mode (no auth)   |
|   Simulator       -->     RAGEngine                                       |
|                           IndustrialAgent (ReAct)   FastAPI               |
|                           ComplianceEngine          REST + WebSocket      |
|                           FeedbackEngine            /api/docs             |
|                                                                          |
|   ROS2 Closed Loop                                                       |
|   -----------------                                                      |
|   VibrationPublisher -> InferenceNode -> EconomicNode -> ActionNode      |
|                                              ^                           |
|                                       HumanOverride (EU Art. 14)        |
|                                                                          |
|   Rigorous Validation Layer (new)                                        |
|   --------------------------------                                       |
|   RealisticFederatedScenario   SiteEconomicProfile calibration           |
|   OperatorSimulationModel      DataSourceMode (explicit hardware flags)  |
|   Network fault injection      Documentation honesty tests               |
|                                                                          |
|   Infrastructure                                                         |
|   --------------                                                         |
|   Docker + Compose   PostgreSQL / SQLite   Celery + Redis                |
|   Prometheus + Grafana   GitHub Actions CI/CD   Terraform (EKS branch)  |
+--------------------------------------------------------------------------+
```

---

## Key Features

| Layer | Feature | Detail |
|-------|---------|--------|
| **AI** | Anomaly Detection | IsolationForest — AUC-ROC >= 0.91 on AI4I 2020 |
| **AI** | Predictive Maintenance | GradientBoosting — 6 failure modes + RUL, MAE <= 18 cycles |
| **AI** | Drift Detection | KS test + PSI + Page-Hinkley — three-layer monitoring |
| **AI** | RAG Document Q&A | FAISS + sentence-transformers + Groq LLM |
| **AI** | Agentic Diagnosis | ReAct tool-calling agent — natural language -> diagnosis |
| **Robotics** | ROS2 Closed Loop | Sensor -> AI -> Economic decision -> actuator -> human override |
| **Robotics** | No-ROS2 Mode | Full pipeline in pure Python — runs in CI, runs in demo |
| **Platform** | Multi-tenancy | Schema-isolated per SME — data cannot cross tenant boundaries |
| **Platform** | Auth | JWT HS256, RBAC (admin / engineer / viewer), refresh token rotation |
| **Platform** | EU AI Act | Article 52 compliance — audit log, transparency report, oversight flag |
| **Platform** | Dashboard | 10-page Streamlit HMI, dark theme, demo mode, real-time charts |
| **Platform** | Workers | Celery + Redis — async retraining, drift checks, cleanup |
| **Rigor** | Non-IID Federation | `RealisticFederatedScenario` — non-uniform data, node dropout |
| **Rigor** | Economic Calibration | `SiteEconomicProfile` — per-site interview, sensitivity analysis |
| **Rigor** | Honest Simulation | `OperatorSimulationModel` — all assumptions cited or flagged |
| **Rigor** | Hardware Mode Flag | `DataSourceMode` enum — no silent hardware/simulation fallback |
| **Ops** | Observability | Prometheus metrics + Grafana dashboards provisioned from config |
| **Ops** | CI/CD | GitHub Actions — lint, type check, 500+ tests, integrity gate |
| **Ops** | Infrastructure | Docker Compose (dev + prod) · Terraform + AWS EKS (experimental) |

---

## ML System

### Anomaly Detection

Model: `IsolationForest` with `StandardScaler` preprocessing
Input: 5-axis sensor readings — temperature, rotational speed, torque, tool wear
Output: `{ label, confidence, anomaly_score, explanation }`

| Dataset | AUC-ROC | F1 (macro) | Min. Threshold |
|---------|---------|------------|----------------|
| AI4I 2020 | >= 0.91 | >= 0.82 | 0.85 / 0.75 |

### Predictive Maintenance + RUL

Model: `GradientBoosting` classifier (6 failure modes) + regressor (remaining useful life)
Output: `{ label, confidence, failure_probability, rul_cycles }`

| Dataset | Metric | Min. Threshold |
|---------|--------|----------------|
| AI4I 2020 | Accuracy | >= 0.85 |
| NASA CMAPSS FD001 | MAE (cycles) | <= 25 |
| NASA CMAPSS FD001 | R2 | >= 0.70 |

### Three-Layer Drift Detection

```
Layer 1 -- KS test        feature-level statistical drift (p < 0.05)
Layer 2 -- PSI            population stability index  (> 0.2 = alert, > 0.25 = retrain)
Layer 3 -- Page-Hinkley   online sequential changepoint detection
```

When PSI exceeds 0.25 or operator feedback accumulates beyond 50 samples, retraining queues automatically via Celery.

### Agentic RAG (ReAct)

```
User:  "Is machine M-003 about to fail?"

Agent: -> classify intent
       -> search_knowledge_base()      FAISS over maintenance manuals
       -> run_anomaly_detection()      real-time sensor scoring
       -> calculate_rul()              remaining useful life estimate
       -> assess_compliance()          EU AI Act Article 52 check
       -> synthesise answer + tool trace + confidence score
       -> set requires_human_review    if confidence < threshold (Art. 14)
```

---

## ROS2 Closed-Loop Pipeline

Most AI platforms stop at "publish a prediction." HAIIP closes the loop.

```
[VibrationPublisher]  sensor_msgs/Imu @ 50 Hz
        |  /haiip/vibration/{machine_id}
        v
[InferenceNode]       POST /api/v1/predict every N samples
        |  /haiip/ai/{machine_id}   label · confidence · anomaly_score
        v
[EconomicNode]        Expected Loss Minimization  (< 1 ms, in-process)
        |  /haiip/decision/{machine_id}   REPAIR_NOW · SCHEDULE · MONITOR · IGNORE
        v
[ActionNode]          decision -> machine command
        |  /haiip/command/{machine_id}   STOP · SLOW_DOWN · MONITOR · NOMINAL
        ^
[HumanOverride]       operator console -- EU AI Act Art. 14
                      override auto-expires (TTL), AI loop resumes
```

**Run without ROS2 installed:**

```bash
# Offline demo -- synthetic vibration, no API call
python -m haiip.ros2.pipeline --no-api

# Live -- with running HAIIP API
python -m haiip.ros2.pipeline

# Fault injection -- bearing defect signature
python -m haiip.ros2.pipeline --fault --machine pump-01

# Human override interactive console
#   s = STOP   d = SLOW_DOWN   m = MONITOR   r = RELEASE
```

**Run with ROS2:**

```bash
ros2 launch haiip haiip_closed_loop.launch.py machine_id:=pump-01
ros2 launch haiip haiip_closed_loop.launch.py fault_mode:=true
```

---

## Research Questions (RQ1–RQ8)

### Core (main branch)

| RQ | Question | Key Metric | Target |
|----|----------|-----------|--------|
| **RQ1** | Can IsolationForest + GradientBoosting achieve >=85% F1 on Nordic SME vibration data? | F1-macro | >= 0.85 |
| **RQ2** | What minimum human oversight rate (HIR) satisfies EU AI Act Article 14 without degrading throughput? | HIR | 0.05–0.15 |
| **RQ3** | Does RAG-based querying reduce engineer time-to-answer vs manual document search? | Latency + groundedness | < 30 s, < 5% hallucination |
| **RQ4** | What is the privacy/compliance trade-off of SHA-256 hashing in audit logs? | Overhead + re-id risk | < 0.1 ms, zero re-id |

### Experimental branch

| RQ | Question | Key Metric | Target |
|----|----------|-----------|--------|
| **RQ5** | How much downtime cost does Expected Loss Minimization avoid vs naive probability thresholding? | False positive rate reduction | 30–50% fewer unnecessary stops |
| **RQ6** | Can FedAvg across 3 Nordic SME nodes achieve F1 within 15% of the centralised baseline while preserving privacy? | `federated_gap` | < 0.15 |
| **RQ7** | What are the measurable Human Override Gain (HOG) and Trust Calibration Score (TCS)? | HOG, TCS | HOG > 0.02, TCS >= 0.80 |
| **RQ8** | What is the compute cost per prediction vs avoided downtime ROI at fleet scale? | ROI ratio | > 10,000x |

**RQ5 formula (ELM vs naive threshold):**
```
E[Cost_wait]   = P(failure) * C_downtime * safety_factor
E[Cost_action] = P(no_failure) * C_false_positive + C_maintenance
Net_benefit    = E[Cost_wait] - E[Cost_action]

Decision thresholds:  REPAIR_NOW (P >= 0.75)  SCHEDULE (P >= 0.50)
                      MONITOR (score >= 0.20)  IGNORE (below noise floor)
```

**RQ6 federated nodes:**

| Node | Country | Industry | Samples | Failure Rate |
|------|---------|----------|---------|--------------|
| SME_FI | Finland | Paper mill | 800 | 12% |
| SME_SE | Sweden | Auto stamping | 1200 | 8% |
| SME_NO | Norway | Offshore pumps | 600 | 18% |

**RQ7 oversight metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| HIR (Human Intervention Rate) | `reviewed / decisions` | 0.05–0.15 |
| HOG (Human Override Gain) | `F1(corrected) - F1(ai_only)` | > 0.02 |
| TCS (Trust Calibration Score) | `1 - ECE` | >= 0.80 |
| ECE (Expected Calibration Error) | Guo et al. (2017) | < 0.10 |

---

## Rigorous Improvement Layer

The platform includes an explicit engineering layer that makes every assumption testable and every failure loud. This is **honest engineering** — not hiding weaknesses but documenting and guarding them.

### Non-IID Federated Scenario (`haiip/core/federated_realistic.py`)

Real Nordic SME nodes have different machine types, failure distributions, and connectivity:

```python
scenario = RealisticFederatedScenario(seed=42)
partitions = scenario.generate_partitions(df)
violations = scenario.get_assumption_violations(partitions["jakobstad"], "jakobstad")
# Returns: ['Jakobstad HDF ratio 0.31, expected 0.60 +/- 0.10']
# Experiment LOGS violations — does NOT silently proceed
```

Node connectivity: each round each node has `p=0.15` chance of dropout, reproducible via seed.

### Economic Parameter Calibration (`haiip/core/economic_calibration.py`)

Default cost parameters are wrong for every real site. Calibrate before quoting any EUR figure:

```python
profile = SiteEconomicProfile()
violations = profile.validate()          # checks all parameter ranges
interview  = profile.calibration_interview()   # 15-min operator questionnaire
profile    = SiteEconomicProfile.from_interview_responses(responses)
df         = profile.sensitivity_analysis()    # which parameters drive the decision?
```

All defaults cite their source. Any parameter outside valid range is an error, not a warning.

### Honest Operator Simulation (`haiip/core/oversight_simulation.py`)

HOG/TCS are computed on simulated operator decisions. Every simulation assumption is named, cited, and confidence-rated:

```
ASSUMPTION_ACCEPT_RATE_EXPERT  = 0.82   # Source: Kaasinen et al. 2022  CONFIDENCE: MEDIUM
ASSUMPTION_FATIGUE_FACTOR      = 0.91   # Source: general HCI literature  CONFIDENCE: LOW
ASSUMPTION_EXPLANATION_BOOST   = 0.08   # Source: model assumption         CONFIDENCE: NONE
```

Any oversight report automatically includes `simulation_confidence: LOW` and `field_study_required: True` until real operator data is collected.

### Hardware Mode Flag (`haiip/data/ingestion/opcua_connector.py`)

No silent fallback from hardware to simulation:

```python
class DataSourceMode(str, Enum):
    REAL_HARDWARE    = "real_hardware"
    SIMULATION       = "simulation"
    HARDWARE_FALLBACK = "hardware_fallback"   # tried hardware, fell back
```

Dashboard shows a visible banner for every mode. `connector.assert_real_hardware()` raises `HardwareNotConnectedError` if not truly connected — call it at the start of any test that claims real hardware.

---

## EU AI Act Compliance

HAIIP is classified **Limited Risk** under EU AI Act Article 52.

| Article | Requirement | Implementation | Status |
|---------|-------------|---------------|--------|
| Art. 9 | Risk classification | `ComplianceEngine.classify_risk()` | OK |
| Art. 12 | Record keeping | `AuditLog` — every prediction, 5-year retention | OK |
| Art. 13 | Information provision | Model Card + Dataset Card + in-app disclosure | OK |
| Art. 14 | Human oversight | `requires_human_review` flag on every prediction | OK |
| Art. 52 | Transparency | Monthly `TransparencyReport` — auto-generated | OK |
| Art. 73 | Incident reporting | `detect_incidents()` — confidence + bias flags | OK |
| GDPR Art. 5c | Data minimisation | Input features SHA-256 hashed in audit log | OK |

The audit log is append-only and exportable as JSON or CSV for regulatory review.

---

## Known Limitations

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the full honest limitations document. Summary:

| Limitation | Severity | Guard |
|-----------|----------|-------|
| **L1** Federated: IID split, all nodes always available | HIGH for research claims | `RealisticFederatedScenario.get_assumption_violations()` |
| **L2** Economic: defaults not calibrated to any real site | HIGH for financial reporting | `SiteEconomicProfile.validate()` + calibration interview |
| **L3** Oversight: HOG/TCS from simulated operator data | HIGH for academic publication | `simulation_confidence: LOW` hardcoded in all reports |
| **L4** Hardware: OPC UA tested against simulator only | MEDIUM for deployment | `DataSourceMode` + `assert_real_hardware()` |

---

## Quick Start

**Prerequisites:** Python 3.11+, [Groq API key](https://console.groq.com) (free)

```bash
git clone https://github.com/Aliipou/HAIIP.git
cd HAIIP
pip install -e ".[dev]"
cp .env.example .env.local
# Set GROQ_API_KEY and SECRET_KEY in .env.local
```

```bash
# Terminal 1 -- API
PYTHONPATH=. uvicorn haiip.api.main:app --port 8000 --reload

# Terminal 2 -- Dashboard
PYTHONPATH=. streamlit run haiip/dashboard/app.py --server.port 8501
```

Open **http://localhost:8501** -- click **Try Demo** (no credentials needed).

```
Demo credentials (local):
  Tenant:   demo-sme
  Email:    admin@haiip.ai
  Password: Demo1234!
```

---

## Docker

```bash
# Development -- SQLite, hot reload, volume mounts
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production -- PostgreSQL, Redis, Prometheus, Grafana
docker compose up
```

| Service | URL | Purpose |
|---------|-----|---------|
| Dashboard | http://localhost:8501 | Streamlit HMI |
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/api/docs | Interactive docs |
| Prometheus | http://localhost:9090 | Metrics |
| Grafana | http://localhost:3000 | Dashboards |

---

## Dashboard -- 10 Pages

| Page | Purpose |
|------|---------|
| Overview | Fleet health, KPI cards, alert summary |
| Live Monitor | Per-machine time series, anomaly history |
| Maintenance | Maintenance scheduler, RUL estimates |
| Alerts | Alert management — acknowledge, filter by severity |
| RAG Q&A | Ask questions about maintenance documentation |
| AI Agent | Interactive ReAct agent — natural language diagnosis |
| Feedback | Submit corrections to improve the model |
| ROI Calculator | Economic decision simulator per site profile |
| Audit Trail | EU AI Act audit log + transparency report export |
| Admin | User management, model registry, tenant stats |

---

## API Reference

### Authentication
```
POST /api/v1/auth/login       ->  access_token + refresh_token
POST /api/v1/auth/refresh     ->  new access_token
POST /api/v1/auth/register    ->  create user (admin only)
GET  /api/v1/auth/me          ->  current user profile
```

### Core Prediction
```
POST /api/v1/predict          ->  sensor reading -> anomaly result + explanation
POST /api/v1/predict/batch    ->  up to 100 readings
GET  /api/v1/predictions      ->  paginated history
GET  /api/v1/predictions/{id} ->  single prediction detail
```

### Human-in-the-Loop
```
GET   /api/v1/alerts                   ->  list alerts (filter by severity)
PATCH /api/v1/alerts/{id}/acknowledge  ->  acknowledge alert
POST  /api/v1/feedback                 ->  submit correction (triggers retraining check)
```

### RAG & Agent
```
POST /api/v1/query             ->  question -> answer + sources + confidence
POST /api/v1/documents/ingest  ->  add document to knowledge base
POST /api/v1/agent/query       ->  natural language -> ReAct diagnosis
POST /api/v1/agent/diagnose    ->  machine_id + readings -> full diagnosis
```

### Observability
```
GET /api/v1/metrics/health     ->  system health KPIs
GET /api/v1/metrics/machines   ->  per-machine statistics
GET /api/v1/audit              ->  EU AI Act audit log (exportable CSV/JSON)
GET /metrics                   ->  Prometheus scrape endpoint
```

**Quick example:**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"tenant_slug":"demo-sme","email":"admin@haiip.ai","password":"Demo1234!"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"machine_id":"pump-01","air_temperature":298.1,"process_temperature":308.6,
       "rotational_speed":1551,"torque":42.8,"tool_wear":0}'
```

---

## Test Suite

```bash
pytest haiip/tests/ -v --cov=haiip --cov-report=term-missing

# By category
pytest haiip/tests/core/              -v   # unit -- ML, RAG, compliance, drift
pytest haiip/tests/api/               -v   # routes, auth, RBAC, pagination
pytest haiip/tests/integration/       -v   # sensor -> prediction -> alert pipelines
pytest haiip/tests/security/          -v   # OWASP Top 10
pytest haiip/tests/crash/             -v   # NaN/Inf, concurrency, empty data
pytest haiip/tests/features/          -v   # BDD role-based user journeys
pytest haiip/tests/robustness/        -v   # network faults, data quality guards
pytest haiip/tests/core/test_ml_rigor.py   # determinism, calibration, fairness
pytest haiip/tests/test_documentation_honesty.py  # no pending results with numbers

# Load test (requires running API)
locust -f haiip/tests/load/locustfile.py \
  --host=http://localhost:8000 --headless --users 50 --spawn-rate 5 --run-time 60s
```

| Category | Count | What It Covers |
|----------|-------|---------------|
| Core unit | ~200 | ML models, RAG, compliance, drift, feedback |
| API unit | ~70 | Routes, auth, RBAC, pagination |
| Integration | ~35 | End-to-end sensor -> prediction -> alert |
| Security | ~30 | OWASP Top 10 — SQL injection, JWT, XSS, RBAC bypass |
| Crash/robustness | ~50 | NaN/Inf inputs, concurrency, empty datasets |
| Network faults | ~20 | OPC UA disconnect, MQTT malformed payload, queue bounds |
| ML rigor | ~25 | Determinism, calibration, fairness across product types |
| Economic calibration | ~15 | Parameter validation, sensitivity analysis, interview round-trip |
| Oversight simulation | ~15 | Assumption citations, confidence reporting, determinism |
| Feature/BDD | ~40 | Role-based user journeys |
| Documentation honesty | ~10 | No pending results with numbers, simulation flags consistent |
| Load (Locust) | 4 user types | SLA: P95 < 500 ms at 50 concurrent users |

---

## CI Pipeline

| Job | Triggers | Purpose |
|-----|---------|---------|
| lint | every push | Ruff + mypy --strict |
| security | every push | Bandit + detect-secrets + Safety |
| unit-tests | every push | Core ML, API routes |
| integration-tests | every push | End-to-end pipeline |
| security-tests | every push | OWASP Top 10 |
| robustness-tests | every push | Network faults, queue bounds |
| ml-rigor-tests | every push | Determinism, calibration, fairness |
| economic-calibration-tests | every push | Parameter ranges, sensitivity |
| oversight-simulation-tests | every push | Assumption citations, confidence |
| **integrity gate** | every PR | Documentation honesty + simulation flags — blocks merge if failed |
| coverage | every push | Aggregated report (target >= 70%) |

---

## Experimental Branch

`git checkout experimental` adds Phase 6 research features on top of everything in `main`:

| Capability | `main` | `experimental` |
|------------|--------|----------------|
| All core AI + ROS2 | OK | OK |
| Economic Decision Engine | -- | OK Expected Loss Minimization |
| Federated Learning | -- | OK FedAvg across 3 Nordic SME nodes |
| Realistic Non-IID Federation | -- | OK RealisticFederatedScenario + dropout |
| Economic Site Calibration | -- | OK SiteEconomicProfile + interview |
| Human Oversight Metrics | qualitative | OK HIR / HOG / TCS quantified |
| Honest Simulation Layer | -- | OK OperatorSimulationModel with citations |
| OpenTelemetry Tracing | -- | OK Distributed tracing + cost model |
| Kubernetes / Helm | -- | OK AWS EKS + HPA |
| Terraform IaC | -- | OK EKS + RDS + ElastiCache |
| Research Notebooks | -- | OK 2 reproducibility notebooks |

```bash
git checkout experimental

# Economic decision engine
python -c "
from haiip.core.economic_ai import EconomicDecisionEngine
d = EconomicDecisionEngine().decide(anomaly_score=0.85, failure_probability=0.80)
print(d.action, f'net benefit: EUR {d.net_benefit:.0f}')
"

# Non-IID federated scenario
python -c "
from haiip.core.federated_realistic import RealisticFederatedScenario
import pandas as pd
scenario = RealisticFederatedScenario(seed=42)
violations = scenario.get_assumption_violations(pd.DataFrame(), 'jakobstad')
print('Assumption violations:', violations)
"

# Site economic calibration
python -c "
from haiip.core.economic_calibration import SiteEconomicProfile
p = SiteEconomicProfile()
print(p.validate())           # [] = all in range
df = p.sensitivity_analysis() # which parameters drive the decision
print(df)
"

# Federated learning across 3 SME nodes
python -c "
from haiip.core.federated import FederatedLearner
r = FederatedLearner().run(n_rounds=10, local_epochs=3)
print(f'Federated F1: {r.final_global_f1:.4f}  gap: {r.federated_gap:+.4f}')
"
```

---

## Production Hardening

HAIIP ships with a security and production layer ready for industrial deployment. Every item below is implemented — not planned.

### Security

| Control | Implementation | Location |
|---------|---------------|----------|
| Rate limiting | Per-IP sliding window — 10 logins/min, 60 predictions/min, 20 agent/min | `haiip/api/middleware.py` |
| Security headers | HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Permissions-Policy | `haiip/api/middleware.py` |
| Body size limit | 10 MB hard cap — rejects oversized requests before route handler | `haiip/api/middleware.py` |
| RBAC | Three roles: admin / engineer / viewer — enforced at every route | `haiip/api/deps.py` |
| JWT rotation | 30 min access + 7 day refresh, HS256 | `haiip/api/auth.py` |
| PII scrubbing | Passwords, tokens, emails → `[REDACTED]` in all structured logs | `haiip/api/middleware.py` |
| IP privacy | Client IPs SHA-256 hashed before logging (GDPR minimisation) | `haiip/api/middleware.py` |
| Secrets | Never in code — all keys in `.env.local` (gitignored) | `.env.example` |

### NGINX + TLS

Production NGINX config at `nginx/haiip.conf` provides:
- HTTP → HTTPS redirect (301)
- TLS 1.2/1.3 only, OCSP stapling, 2-year HSTS with preload
- Upstream rate limiting mirroring API limits (`limit_req_zone`)
- WebSocket proxying for Streamlit (`/_stcore/stream`)
- `/metrics` restricted to internal network (`10.0.0.0/8`)
- `client_max_body_size 10M` aligned with API middleware

```bash
# Install cert (replace YOUR_DOMAIN)
certbot certonly --nginx -d YOUR_DOMAIN

# Edit nginx/haiip.conf — replace YOUR_DOMAIN
# Deploy
sudo cp nginx/haiip.conf /etc/nginx/sites-enabled/haiip.conf
nginx -t && systemctl reload nginx
```

### SHAP Explainability

Every anomaly prediction now optionally includes SHAP feature attributions alongside z-scores:

```python
detector = AnomalyDetector()
detector.fit(normal_data)
result = detector.predict([298.1, 308.6, 1551, 42.8, 0])

# result["explanation"]  — z-scores for features |z| > 1.5 (always present)
# result["shap_values"]  — SHAP TreeExplainer values (present when shap is installed)
# {
#   "torque": 0.0312,       # SHAP: positive = pushed toward anomaly
#   "tool_wear": -0.0041,
#   ...
# }
```

SHAP is computed with a cached `TreeExplainer` (built once after `fit()`). Batch predictions compute all SHAP values in a single vectorised call. Falls back gracefully to z-score explanation if `shap` is not installed.

---

## Datasets

| Dataset | Source | License | Used For |
|---------|--------|---------|---------|
| AI4I 2020 | UCI ML Repository | CC BY 4.0 | Failure mode classification, anomaly detection |
| NASA CMAPSS | NASA Prognostics COE | Public Domain | Remaining Useful Life prediction |
| CWRU Bearing | Case Western Reserve | Public Domain | Bearing fault detection |
| MIMII (Hitachi) | Zenodo | CC BY 4.0 | Sound-based anomaly -- planned v0.2 |

Full datasheets following Gebru et al. (2021) format: [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md)

---

## RDI Artifacts

| Artifact | Location |
|----------|----------|
| Model Card (Mitchell et al., 2019) | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| Dataset Card (Gebru et al., 2021) | [`docs/DATASET_CARD.md`](docs/DATASET_CARD.md) |
| Honest Limitations Document | [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) |
| Results Register | [`docs/RESULTS.md`](docs/RESULTS.md) |
| EU AI Act compliance engine | [`haiip/core/compliance.py`](haiip/core/compliance.py) |
| Non-IID federated scenario | [`haiip/core/federated_realistic.py`](haiip/core/federated_realistic.py) |
| Economic site calibration | [`haiip/core/economic_calibration.py`](haiip/core/economic_calibration.py) |
| Operator simulation model | [`haiip/core/oversight_simulation.py`](haiip/core/oversight_simulation.py) |
| Transparency report generator | `ComplianceEngine.generate_transparency_report()` |
| Security tests (OWASP Top 10) | [`haiip/tests/security/`](haiip/tests/security/) |
| Network fault injection tests | [`haiip/tests/robustness/`](haiip/tests/robustness/) |
| ML rigor tests | [`haiip/tests/core/test_ml_rigor.py`](haiip/tests/core/test_ml_rigor.py) |
| Documentation honesty tests | [`haiip/tests/test_documentation_honesty.py`](haiip/tests/test_documentation_honesty.py) |
| RAG hallucination test suite | [`haiip/tests/core/test_rag_hallucination.py`](haiip/tests/core/test_rag_hallucination.py) |
| ML evaluation benchmarks | [`haiip/tests/features/test_ml_evaluation.py`](haiip/tests/features/test_ml_evaluation.py) |
| Load tests | [`haiip/tests/load/locustfile.py`](haiip/tests/load/locustfile.py) |
| CI/CD pipeline | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| Integrity gate | [`.github/workflows/integrity.yml`](.github/workflows/integrity.yml) |

---

## Citation

```bibtex
@techreport{haiip2025,
  title       = {HAIIP: Human-Aligned Industrial Intelligence Platform},
  author      = {Pourrahim, Ali},
  year        = {2025},
  institution = {Centria University of Applied Sciences},
  note        = {RDI deliverable -- NextIndustriAI project
                 (Jakobstad / Sundsvall / Narvik).
                 Features: predictive maintenance, ROS2 closed-loop automation,
                 EU AI Act compliance, multi-tenant SaaS, agentic RAG.
                 Experimental branch: federated learning, economic AI,
                 quantified human oversight metrics, rigorous validation layer.}
}
```

---

<div align="center">

HAIIP is a research prototype built as an RDI deliverable.
Not a certified safety-critical device. Production deployment requires site-specific calibration and field validation.

**[Demo video](https://1drv.ms/v/c/5978203504f409d5/IQALgoMRTYgqT6l8gsL6-nnQAXSjb32DH33UhkqUrdpd3fA?e=sneUNO) · [github.com/Aliipou/HAIIP](https://github.com/Aliipou/HAIIP)**

</div>
