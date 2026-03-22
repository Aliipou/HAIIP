# Model Card — HAIIP AI Models

## Overview

HAIIP uses an ensemble of models for industrial anomaly detection and predictive maintenance.

| Model | Type | Task |
|---|---|---|
| AnomalyDetector | Isolation Forest | Unsupervised anomaly detection |
| AnomalyAutoencoder | LSTM Autoencoder | Time-series reconstruction error |
| MaintenancePredictor | Gradient Boosting | Binary failure classification |
| MaintenanceLSTM | Bidirectional LSTM | Sequential failure prediction |
| DriftDetector | Statistical (KS-test) | Concept/data drift monitoring |

## Intended Use

**Intended users:** Industrial maintenance engineers and operators at Nordic SMEs.

**Intended use:** Real-time machine health monitoring, predictive maintenance scheduling.

**Out-of-scope:** Safety-critical control systems (aviation, nuclear). Medical devices.

## Performance

| Metric | AnomalyDetector | MaintenancePredictor |
|---|---|---|
| Precision | 0.87 | 0.91 |
| Recall | 0.83 | 0.88 |
| F1 | 0.85 | 0.89 |

## Limitations

- Models trained primarily on rotating machinery (motors, pumps, compressors)
- Requires minimum 72 hours of baseline data
- Drift detection may trigger false positives during planned maintenance

## EU AI Act Classification

**Risk level:** Limited Risk (Article 52)

Every AI decision includes a plain-language explanation. Operator override is always available. All predictions are logged with confidence scores and input features.
