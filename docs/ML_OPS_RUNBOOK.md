# HAIIP ML Ops Runbook

> Operational guide for the machine learning pipeline.
> Covers: model training, ONNX export, auto-retraining, latency monitoring, rollback.

---

## Overview

HAIIP uses a two-tier model stack:

| Tier | Models | Where |
|------|--------|-------|
| **Sklearn** | `AnomalyDetector` (IsolationForest), `MaintenancePredictor` (GradientBoosting) | API server + Celery workers |
| **Deep (Lightning)** | `AnomalyAutoencoder` (LSTM), `MaintenanceLSTM` (BiLSTM) | API server / training only |
| **Edge** | `ONNXAnomalyDetector`, `ONNXMaintenancePredictor` | Edge nodes (Jetson / Industrial PC) |

The **AutoRetrainPipeline** connects all three: drift triggers retraining of the sklearn champion,
a challenger is evaluated, and the winner is exported to ONNX for the edge.

---

## Directory Structure

```
/artifacts/
  {tenant_id}/
    anomaly/                    sklearn AnomalyDetector (joblib)
      scaler.joblib
      isolation_forest.joblib
    autoencoder/                Lightning AnomalyAutoencoder
      autoencoder.pt
      meta.npz
    lstm_maintenance/           Lightning MaintenanceLSTM
      lstm_maintenance.pt
      meta.npz
      classes.json
    drift_reference.npy         reference distribution (training data)
    drift_current.npy           rolling current window
    onnx/
      anomaly_autoencoder.onnx  edge-ready ONNX model
      maintenance_lstm.onnx
```

---

## Daily Operations

### 1. Check pipeline status

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/ml-ops/pipeline-status?tenant_id=demo-sme
```

Expected response:
```json
{
  "models": {
    "sklearn_champion": true,
    "pytorch_autoencoder": false,
    "onnx_anomaly": true
  },
  "drift": { "detected": false, "severity": "stable" },
  "sla_target_ms": 50
}
```

If `drift.detected` is `true` and `severity` is `"drift"`, trigger retraining manually.

### 2. Check Celery beat tasks

```bash
# Check Celery beat is running
celery -A haiip.workers.tasks inspect scheduled

# Force a drift check immediately
celery -A haiip.workers.tasks call haiip.workers.tasks.run_drift_check
```

### 3. Monitor latency

```bash
# Benchmark ONNX model (200 runs)
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "demo-sme", "model_type": "anomaly", "n_runs": 200}' \
  http://localhost:8000/api/v1/ml-ops/benchmark
```

Check `sla_pass` in the response. If `sla_pass: false` or `p99_ms > 50`, see
[SLA Breach Response](#sla-breach-response) below.

---

## Retraining

### Automatic (every 6 hours via Celery beat)

The `auto_retrain_pipeline` task runs every 6 hours. It:
1. Loads the drift reference + current data
2. Runs `DriftDetector.check()`
3. Fires `AutoRetrainPipeline.maybe_retrain()` if triggers are met
4. Promotes challenger if F1 improves by ≥ 0.01
5. Saves the new champion to `/artifacts/{tenant_id}/anomaly/`

No intervention needed if working correctly.

### Manual trigger (API)

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "demo-sme", "force_reason": "manual"}' \
  http://localhost:8000/api/v1/ml-ops/retrain
```

### Manual trigger (Celery)

```bash
celery -A haiip.workers.tasks call haiip.workers.tasks.auto_retrain_pipeline \
  --args='["demo-sme"]' --kwargs='{"force_reason": "manual"}'
```

### After hardware maintenance / tool change

After a major machine event, retrain immediately:

```python
from haiip.core.auto_retrain import AutoRetrainPipeline, TriggerReason

pipeline = AutoRetrainPipeline(tenant_id="sme-fi")
# ... register champion ...
event = pipeline.maybe_retrain(X_new, reason=TriggerReason.MANUAL)
print(event.promoted, event.challenger_metrics)
```

---

## ONNX Export

### Export to ONNX (API)

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "demo-sme", "model_type": "anomaly", "opset": 17}' \
  http://localhost:8000/api/v1/ml-ops/export-onnx
```

### Export to ONNX (Python)

```python
from haiip.core.torch_models import AnomalyAutoencoder

model = AnomalyAutoencoder.load("artifacts/demo-sme/autoencoder")
onnx_path = model.export_onnx("artifacts/demo-sme/onnx/anomaly_autoencoder.onnx")
print(f"Exported: {onnx_path} ({onnx_path.stat().st_size // 1024} KB)")
```

### Deploy to edge node

```bash
scp artifacts/demo-sme/onnx/anomaly_autoencoder.onnx jetson-node:/opt/haiip/models/
scp artifacts/demo-sme/autoencoder/meta.npz jetson-node:/opt/haiip/models/
```

On the edge node:

```python
from haiip.core.onnx_runtime import ONNXAnomalyDetector

detector = ONNXAnomalyDetector.from_onnx("/opt/haiip/models/anomaly_autoencoder.onnx")
result = detector.predict([298.1, 308.6, 1551, 42.8, 0])
# → {"label": "normal", "latency_ms": 3.8, "sla_ok": True}
```

---

## Champion-Challenger Logic

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_improvement` | 0.01 | Minimum F1 gain to promote challenger |
| `auc_tolerance` | 0.02 | Maximum AUC regression allowed on promotion |
| `cooldown_minutes` | 60 | Minimum minutes between retrain triggers |
| `drift_feature_threshold` | 2 | Minimum features in "drift" state to trigger |
| `accuracy_threshold` | 0.80 | Accuracy below this fires retraining |

**Promotion rule:**

```
promote if:
  challenger.f1_macro >= champion.f1_macro + min_improvement
  AND
  champion.auc_roc - challenger.auc_roc <= auc_tolerance
```

If promotion is rejected, the challenger is discarded and the champion remains in production.

**Audit trail**: every retrain cycle produces a `RetrainEvent` with:
- `trigger_reason` (drift_critical / accuracy_drop / scheduled / manual)
- `champion_metrics` and `challenger_metrics` (F1, AUC, n_samples)
- `promoted` (bool)
- `triggered_at` and `completed_at` (ISO 8601)

---

## SLA Breach Response

Target: **p99 ≤ 50 ms** on CPU.

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| p99 > 50ms on CPU | Model too large for hardware | Reduce `hidden_size` or `n_layers`, retrain, re-export |
| p99 > 50ms on GPU | GPU provider not enabled | Check `onnxruntime-gpu` is installed, CUDA available |
| SLA breach on batch | Batch size too large | Reduce batch size or use streaming inference |
| Session None / not ready | ONNX file missing | Re-run export-onnx task |

---

## Rollback

If the promoted champion regresses in production:

1. Identify the previous champion artifact:

```bash
ls -lt artifacts/demo-sme/anomaly/
# The most recently modified .joblib files = current champion
# git log on artifact dir if version-controlled
```

2. Restore manually:

```python
from haiip.core.anomaly import AnomalyDetector

old_champion = AnomalyDetector.load("artifacts/demo-sme/anomaly_backup/")
old_champion.save("artifacts/demo-sme/anomaly/")
```

3. Re-export to ONNX:

```bash
curl -X POST ... /api/v1/ml-ops/export-onnx
```

**Recommendation**: Keep a dated backup copy before every promotion.

```bash
cp -r artifacts/demo-sme/anomaly artifacts/demo-sme/anomaly_$(date +%Y%m%d_%H%M)
```

---

## Drift Reference Update

When a machine is overhauled or operating conditions change permanently, update the reference distribution:

```python
import numpy as np

X_new_normal = ...  # post-overhaul normal data
np.save("artifacts/demo-sme/drift_reference.npy", X_new_normal)
```

Then trigger a retrain to align the champion with the new distribution.

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| `auto_retrain_pipeline` always says `no_retrain_needed` | Check cooldown: `trigger.reset_cooldown()` or wait 60 min |
| Challenger never promotes | `min_improvement` too high, or validation set too small — verify metrics in `RetrainEvent` |
| ONNX export fails | Torch/onnx not installed, or model not fitted — check logs |
| `benchmark` returns `skipped` | ONNX file does not exist at expected path — run export first |
| Celery beat not running | `celery -A haiip.workers.tasks beat --loglevel=info` |
| Redis not reachable | Check `CELERY_BROKER_URL` in `.env.local` |
