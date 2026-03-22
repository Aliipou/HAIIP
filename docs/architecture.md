# Architecture

## System Layers

### 1. Data Ingestion

- **OPC UA** (asyncua) — PLCs, SCADA
- **MQTT** (paho-mqtt) — IoT sensors
- **CSV** — Vibration analyzer exports
- **Simulator** — Synthetic data for demos

### 2. Feature Engineering

Rolling window features computed in real time:

| Feature | Window | Purpose |
|---|---|---|
| mean | 60s / 300s / 3600s | Trend detection |
| std | 60s / 300s | Variability |
| min / max | 300s | Range tracking |
| spectral entropy | 60s | Vibration signature |

### 3. AI Core

**Anomaly Detection (unsupervised):**
- Isolation Forest — fast, interpretable, CPU inference
- LSTM Autoencoder — catches temporal patterns

**Predictive Maintenance (supervised):**
- Gradient Boosting — high precision, SHAP explainability
- BiLSTM — bidirectional temporal dependencies

### 4. Human-in-the-Loop

Every prediction is shown to the operator with:
- Plain-language explanation
- Confidence score
- Suggested action
- Override button (always active, always logged)

### 5. Compliance Engine

Append-only audit log for every prediction:

```python
@dataclass
class AuditRecord:
    timestamp: datetime
    model_version: str
    input_features: dict[str, float]
    prediction: float
    confidence: float
    explanation: str
    operator_action: str | None
    override_reason: str | None
```

## Deployment

```bash
git clone https://github.com/Aliipou/HAIIP.git
cd HAIIP
docker compose up --build
# Dashboard: http://localhost:8501
# API docs:  http://localhost:8000/api/docs
```

**Stack:** Python 3.11, FastAPI, Streamlit, scikit-learn, PyTorch, asyncpg, Redis, Docker Compose
