<div align="center">

# HAIIP

### Human-Aligned Industrial Intelligence Platform

**Production-grade AI for Nordic industrial SMEs — predictive maintenance, anomaly detection, EU AI Act compliant**

[![CI](https://github.com/Aliipou/HAIIP/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/HAIIP/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](pyproject.toml)
[![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act-Limited%20Risk-green)](docs/MODEL_CARD.md)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](LICENSE)

> RDI deliverable 

</div>

---

## The Problem

Nordic industrial SMEs lose **EUR 500–2000 per hour** of unplanned downtime. They cannot afford data scientists. They cannot deploy black-box AI that operators distrust. From August 2026, every AI system must comply with the EU AI Act.

HAIIP solves all three.

---

## What HAIIP Does

| Capability | How |
|---|---|
| **Real-time monitoring** | OPC UA, MQTT, vibration CSV, built-in simulator |
| **Failure prediction** | Gradient Boosting + BiLSTM, explains why in plain language |
| **Human-in-the-loop** | Operator can override every AI decision |
| **Closed-loop control** | Decision reaches the machine actuator, not just a dashboard |
| **EU AI Act compliance** | Full audit trail, model cards, drift detection |
| **RDI reporting** | Generates evidence artifacts for EU-funded projects |

---

## Architecture

```
Data Sources          AI Core                     Interfaces
------------          -------                     ----------
OPC UA (PLC)    -->   AnomalyDetector (IF)        Streamlit HMI (10 pages)
MQTT broker     -->   MaintenancePredictor (GB)   FastAPI REST + WebSocket
Vibration CSV   -->   AnomalyAutoencoder (LSTM)   /api/docs (OpenAPI)
Simulator       -->   MaintenanceLSTM (BiLSTM)    Demo mode (no auth)
                      DriftDetector
                      AutoRetrainPipeline
                      RAGEngine + LLM Agent
                      ComplianceEngine
```

**Stack:** Python 3.11 · FastAPI · Streamlit · scikit-learn · PyTorch · asyncpg · Redis · Docker Compose

---

## Quick Start

```bash
git clone https://github.com/Aliipou/HAIIP.git
cd HAIIP
docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/api/docs
- Demo mode works without real hardware

---

## Project Structure

```
HAIIP/
├── src/
│   ├── ai/              # Models: anomaly detection, predictive maintenance
│   ├── compliance/      # EU AI Act compliance engine
│   ├── data/            # OPC UA, MQTT, CSV ingestion
│   └── api/             # FastAPI application
├── ui/                  # Streamlit HMI
├── docs/
│   ├── MODEL_CARD.md
│   └── DATASET_CARD.md
└── docker-compose.yml
```

---

## EU AI Act Compliance

HAIIP targets **Limited Risk** classification under Article 52:

- Full audit trail for every AI decision
- Model cards and dataset cards
- Drift detection with automatic retraining
- Human override at every decision point
- Explainability layer (SHAP + plain-language summaries)

---

## License

Proprietary — contact the repository owner for licensing inquiries.
